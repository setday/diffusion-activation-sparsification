# Modified from fast-dit's repo: https://github.com/chuanyangjin/fast-DiT/tree/main

import os
import math

from tqdm import tqdm
from PIL import Image

import numpy as np

import torch
import torch.distributed as dist

from diffusers.models import AutoencoderKL

from diffusion import create_diffusion

from utils.train_utils import find_model
from utils.common import setup_env, parse_common_args, create_model


setup_env()


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path



#################################################################################
#                                  Sampling Loop                                #
#################################################################################


def main(args):
    """
    Run sampling.
    """
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    latent_size = args.image_size // 8
    model = create_model(args).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    if args.ckpt:
        state_dict = find_model(args.ckpt)
        model.load_state_dict(state_dict, strict=False)
        
    if args.adapter_ckpt:
        # Load adapter states (e.g. router weights for polar sparsity)
        adapter_state_dict = torch.load(args.adapter_ckpt, map_location=lambda storage, loc: storage)
        if "adapter_model" in adapter_state_dict:
            adapter_state_dict = adapter_state_dict["adapter_model"]
        elif "model" in adapter_state_dict:
            adapter_state_dict = adapter_state_dict["model"]
        elif "ema" in adapter_state_dict:
            adapter_state_dict = adapter_state_dict["ema"]
        model.load_state_dict(adapter_state_dict, strict=False)
        if rank == 0:
            print(f"Loaded adapter checkpoint from {args.adapter_ckpt}")

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    adapter_string = f"-adapter-{os.path.basename(args.adapter_ckpt).replace('.pt', '')}" if args.adapter_ckpt else ""
    folder_name = f"{model_string_name}-{args.act_layer}-{args.norm_layer}-{args.attn_layer}-{args.mlp_layer}-{ckpt_string_name}{adapter_string}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    assert global_batch_size % dist.get_world_size() == 0, "global_batch_size must be divisible by world_size"
    global_batch_size = args.global_batch_size
    n = global_batch_size // dist.get_world_size()  # Per-GPU batch size
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    all_samples = np.zeros((total_samples, args.image_size, args.image_size, 3), dtype=np.uint8)
    for epoch in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        with torch.autocast(device_type=f"cuda:{device}", dtype=torch.bfloat16):
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        all_samples[n * epoch : n * (epoch+1)] = samples

        # Save samples to disk as individual .png files
        if epoch == 0:
            for i, sample in enumerate(samples):
                if i >= 10:
                    break
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        npz_path = f"{sample_folder_dir}.npz"
        np.savez(npz_path, arr_0=all_samples)
        print(f"Saved .npz file to {npz_path} [shape={all_samples.shape}].")
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = parse_common_args(add_model_args=True, add_vae_args=True)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--ckpt", type=str, help="Path to a DiT checkpoint.")
    parser.add_argument("--adapter-ckpt", type=str, default=None, help="Optional path to a DiT adapter checkpoint (e.g. for Polar Sparsity routers).")
    
    args = parser.parse_args()
    main(args)
