import argparse
import torch

def setup_env():
    """Sets up PyTorch environment optimizations."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes


def parse_common_args(
        parser=None,
        add_vae_args=True,
        add_model_args=False,
        add_training_args=False
    ):
    """
    Returns an ArgumentParser with common arguments for models, 
    datasets, and training/sampling logic.
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    
    # Core model args
    from models.dit import DiT_models
    from models.diffit import DiffiT_models
    from modifiers.activation import ACTIVATIONS
    from modifiers.normalization import NORMALIZATIONS
    from modifiers.linear import LINEARS
    from modifiers.attention import ATTENTIONS
    from modifiers.mlp import MLPS
    
    parser.add_argument("--global-seed", type=int, default=0)

    if add_vae_args:
        parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
        parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    
    model_choices = list(DiT_models.keys()) + list(DiffiT_models.keys())
    if add_model_args:
        parser.add_argument("--model", type=str, choices=model_choices, default="DiT-XL/2")
        parser.add_argument("--num-classes", type=int, default=1000)
    
    # Layer norm, activations, and linear layers
    if add_model_args:
        parser.add_argument("--act-layer", type=str, choices=ACTIVATIONS.keys(), default="GELU")
        parser.add_argument("--norm-layer", type=str, choices=NORMALIZATIONS.keys(), default="LayerNorm")
        parser.add_argument("--lin-layer", type=str, choices=LINEARS.keys(), default="Linear")
        parser.add_argument("--attn-layer", type=str, choices=ATTENTIONS.keys(), default="Attention")
        parser.add_argument("--mlp-layer", type=str, choices=MLPS.keys(), default="Mlp")
    
    # Training Adapter Only
    if add_training_args:
        parser.add_argument("--train-adapters-only", action="store_true", help="Freeze base model and only train router adapters")

    # Data/Compute
    parser.add_argument("--global-batch-size", type=int, default=256)
    if add_training_args:
        parser.add_argument("--num-workers", type=int, default=4)
    
    return parser


def create_model(args):
    """
    Creates the DiT or DiffiT model based on provided arguments.
    """
    from models.dit import DiT_models
    from models.diffit import DiffiT_models
    from modifiers.activation import ACTIVATIONS
    from modifiers.normalization import NORMALIZATIONS
    from modifiers.linear import LINEARS
    from modifiers.attention import ATTENTIONS
    from modifiers.mlp import MLPS
    
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    act_layer = ACTIVATIONS[args.act_layer]
    norm_layer = NORMALIZATIONS[args.norm_layer]
    lin_layer = LINEARS[args.lin_layer]
    attn_layer = ATTENTIONS[args.attn_layer]
    mlp_layer = MLPS[args.mlp_layer]

    if args.model in DiT_models:
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            act_layer=act_layer,
            norm_layer=norm_layer,
            lin_layer=lin_layer,
            attn_layer=attn_layer,
            mlp_layer=mlp_layer,
        )
    elif args.model in DiffiT_models:
        model = DiffiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes,
            act_layer=act_layer,
            norm_layer=norm_layer,
            lin_layer=lin_layer,
            attn_layer=attn_layer,
            mlp_layer=mlp_layer,
        )
    else:
        raise ValueError(f"Model {args.model} not found in DiT_models or DiffiT_models.")

    return model
