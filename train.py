'''
SDiT:Spiking diffusion model with transformer
This codebase borrows from DiT (https://github.com/facebookresearch/DiT)
License of DiT:
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import torchvision

from model.mixer import Diffusion_RWKV,RWKV_Config
from diffusion import create_diffusion
from spikingjelly.activation_based import functional

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)




def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger

        root_logger = logging.getLogger()
        for h in root_logger.handlers:
            root_logger.removeHandler(h)
            
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)


    # Create model:
    config = RWKV_Config(ctx_len=768,model_type='RWKV',vthreshold=1.)
    if args.dataset == "MNIST":
        model = Diffusion_RWKV(input_size=28,patch_size=2,in_channels=3,hidden_size=384,depth=4,num_classes=10,config=config) #MNIST
    if args.dataset == "FMNIST":
        model = Diffusion_RWKV(input_size=28,patch_size=2,in_channels=3,hidden_size=384,depth=4,num_classes=10,config=config) #FMNIST
    if args.dataset == "CIFAR10":
        model = Diffusion_RWKV(input_size=28,patch_size=2,in_channels=3,hidden_size=512,depth=8,num_classes=10,config=config) #CIFAR10
 
    if args.resume:
        logger.info(f"resuming...........{args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location = {'cuda:%d' % 0: 'cuda:%d' % rank})
        model.load_state_dict(state_dict['model'])

    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="",diffusion_steps=1000)  # default: 1000 steps, linear noise schedule

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)    
    if args.resume:
        opt.load_state_dict(state_dict['opt'])

    logger.info(f"SDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[400,800],gamma=0.2)
    scheduler = None

    # Setup data:
    MNIST_transform  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
    FMNIST_transform  = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1)),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    CIFAR10_transform  = torchvision.transforms.Compose([torchvision.transforms.Resize(28),torchvision.transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    if args.dataset == "MNIST":
        dataset = torchvision.datasets.MNIST(root="/home/dataset/",train=True,transform=MNIST_transform,download=False)
    if args.dataset == "FMNIST":
        dataset = torchvision.datasets.FashionMNIST(root="/home/dataset/",train=True,transform=FMNIST_transform,download=False)
    if args.dataset == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root="/home/dataset/",train=True,transform=CIFAR10_transform,download=False)


    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,}")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            functional.reset_net(model)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Log loss values:
            running_loss += loss.item()

            log_steps += 1
            train_steps += 1
            # if rank == 0:
            #     print(running_loss/train_steps)
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                running_loss = 0
                log_steps = 0
                start_time = time()

        if (epoch+1) % args.ckpt_every == 0 and epoch > 0 :
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{epoch:05d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        if scheduler is not None:
            scheduler.step()

    model.eval() 


    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--model", type=str, default="SDiT")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--ckpt", type=str, default=None)  
    args = parser.parse_args()
    main(args)
