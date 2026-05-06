# Modified from fast-dit's repo: https://github.com/chuanyangjin/fast-DiT/tree/main

"""
A minimal training script for DiT.
"""
from copy import deepcopy
from time import time
import os

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator

from diffusion import create_diffusion

from utils.train_utils import find_model, update_ema, create_logger
from utils.dataset_utils import FeatureDataset as CustomDataset
from utils.common import setup_env, parse_common_args, create_model


setup_env()


def main(args):
    """
    Trains a new DiT model or ControlNet adapter.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        suffix = "-controlnet" if args.train_controlnet else ""
        experiment_dir = f"{args.results_dir}/{model_string_name}-{args.act_layer}-{args.norm_layer}-{args.attn_layer}-{args.mlp_layer}{suffix}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    model = create_model(args)

    is_finetune = args.ckpt is not None
    if is_finetune:
        state_dict = find_model(args.ckpt)
        model.load_state_dict(state_dict)

    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)

    # Handle ControlNet training
    if args.train_controlnet:
        from models.controlnet import ControlNetAdapter, ControlNetTrainer

        model = ControlNetAdapter(
            base_model=model,
            num_classes=args.num_classes,
            hidden_size=model.hidden_size if hasattr(model, 'hidden_size') else 384,
            control_depth=args.controlnet_depth,
        )
        model = model.to(device)

        # Freeze base model, only train ControlNet
        ControlNetTrainer.freeze_base_model(model)

        if args.controlnet_ckpt is not None:
            ckpt = torch.load(args.controlnet_ckpt, map_location=device)
            model.control_net.load_state_dict(ckpt)

        if accelerator.is_main_process:
            logger.info("ControlNet training mode enabled. Base model frozen.")

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # ----------------------------------------------------
    # FREEZE BASE MODEL, UNFREEZE ADAPTERS IF REQUESTED
    # ----------------------------------------------------
    trainable_params = model.parameters()
    if args.train_adapters_only:
        for p in model.parameters():
            p.requires_grad = False

        trainable_params_list = []
        for name, module in model.named_modules():
            # Check strictly for `.router` attribute, skipping attention projections or base MLPs
            if hasattr(module, 'router_loss') and hasattr(module, 'router'):
                for p in module.router.parameters():
                    p.requires_grad = True
                    trainable_params_list.append(p)

        trainable_params = trainable_params_list

        if accelerator.is_main_process:
            num_trainable = sum(p.numel() for p in trainable_params)
            num_total = sum(p.numel() for p in model.parameters())
            logger.info(f"Adapter Training Mode On! Total Parameters: {num_total:,} | Trainable Parameters: {num_trainable:,}")

        if len(trainable_params) == 0 and accelerator.is_main_process:
            logger.warning("No trainable parameters found! Did you pass appropriate Polar modifiers?")

    for p in ema.parameters():
        p.requires_grad = False
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if accelerator.is_main_process:
        logger.info(f"{args.model} Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.train_controlnet:
        from models.controlnet import ControlNetTrainer
        opt = ControlNetTrainer.create_optimizer(model, lr=1e-4)
        if accelerator.is_main_process:
            logger.info("ControlNet optimizer created for trainable parameters only")
    else:
        lr = 1e-4 if args.model.startswith('DiT') else 3e-4
        weight_decay = 0.0 if args.train_adapters_only else (0.0 if args.model.startswith('DiT') else 0.9999)
        if is_finetune:
            lr /= 10  # Use a smaller learning rate for finetuning
        if accelerator.is_main_process:
            logger.info(f"{lr=} {weight_decay=}")
        opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Setup data:
    dataset = CustomDataset(args.feature_path)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # =======================================================

    if accelerator.is_main_process:
        checkpoint = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/{0:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    # =======================================================

    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            if args.train_adapters_only:
                # Only train the adapters (routers). Do a forward pass to collect router self-loss.
                _ = diffusion.training_losses(model, x, t, model_kwargs)

                total_router_loss = 0.0
                num_routers = 0
                for module in model.modules():
                    if hasattr(module, 'module'): # unwraps DDP
                        module = module.module
                    if hasattr(module, 'router_loss'):
                        total_router_loss += module.router_loss
                        num_routers += 1

                loss = total_router_loss / max(1, num_routers)
            else:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes

                avg_loss = torch.tensor(avg_loss, device=accelerator.device)
                avg_loss = accelerator.reduce(avg_loss, reduction="sum")

                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    if args.train_adapters_only:
                        unwrapped_model = accelerator.unwrap_model(model)
                        adapter_state_dict = {
                            k: v for k, v in unwrapped_model.state_dict().items() if "router" in k
                        }
                        checkpoint = {
                            "adapter_model": adapter_state_dict,
                            "opt": opt.state_dict(),
                            "args": args
                        }
                    elif args.train_controlnet:
                        unwrapped_model = accelerator.unwrap_model(model)
                        controlnet_state = unwrapped_model.control_net.state_dict()
                        injection_state = {
                            k: v for k, v in unwrapped_model.state_dict().items() if "injection" in k
                        }
                        checkpoint = {
                            "control_net": controlnet_state,
                            "injections": injection_state,
                            "opt": opt.state_dict(),
                            "args": args
                        }
                    else:
                        checkpoint = {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }

                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    parser = parse_common_args(add_model_args=True, add_training_args=True)
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=1400) # For ft use 20 epochs.
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a DiT checkpoint for finetuning")
    parser.add_argument("--train-controlnet", action="store_true", help="Train a ControlNet adapter for class conditioning")
    parser.add_argument("--controlnet-ckpt", type=str, default=None, help="Path to pre-trained ControlNet checkpoint")
    parser.add_argument("--controlnet-depth", type=int, default=6, help="Depth of ControlNet")

    args = parser.parse_args()
    main(args)
