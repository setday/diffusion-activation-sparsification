# Modified from fast-dit's repo: https://github.com/chuanyangjin/fast-DiT/tree/main

import os

from tqdm import tqdm

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets import load_dataset
from diffusers.models import AutoencoderKL

from utils.common import setup_env, parse_common_args
from utils.dataset_utils import center_crop_arr, CustomDataset


setup_env()


def main(args):
    """
    Extract features from a pre-trained VAE for the entire ImageNet-1K training set using DDP, and save them to .npy files.
    """
    
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

    # Setup a feature folder:
    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'imagenet256_features'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'imagenet256_labels'), exist_ok=True)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup data:
    local_batch_size = args.global_batch_size // dist.get_world_size()
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = CustomDataset(load_dataset(
        "timm/imagenet-1k-wds", # "pcuenq/lsun-bedrooms"
        split="train",
        trust_remote_code=True,
        num_proc=16,
    ), transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size = local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    xs, ys = [], []
    for x, y in tqdm(loader, total=len(loader), desc=f"Rank {rank}"):
        x = x.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            
        xs.append(x.detach().cpu())  # (bs, 4, 32, 32)
        ys.append(y)                 # (bs,)

    np.save(f'{args.features_path}/imagenet256_features_all.npy', torch.vstack(xs).numpy())
    np.save(f'{args.features_path}/imagenet256_labels_all.npy', torch.vstack(ys).numpy())
    
    dist.destroy_process_group()

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = parse_common_args(add_vae_args=True, add_training_args=True)
    parser.add_argument("--features-path", type=str, default="features")
    args = parser.parse_args()
    main(args)
