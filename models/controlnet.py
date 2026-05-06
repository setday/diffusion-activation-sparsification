"""
ControlNet implementation for class-conditioned diffusion models.
Based on: Zhang et al., 2023, "Adding Conditional Control to Text-to-Image Diffusion Models"
"""

import torch
import torch.nn as nn
from models.common_layers import TimestepEmbedder, LabelEmbedder, Mlp, Attention


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ControlNetBlock(nn.Module):
    """Single ControlNet block for spatial control."""

    def __init__(
        self,
        channels,
        num_heads,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        lin_layer=nn.Linear,
    ):
        super().__init__()
        self.norm = norm_layer(channels, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(channels, num_heads=num_heads, qkv_bias=True, lin_layer=lin_layer)

        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=0,
            lin_layer=lin_layer,
        )

        self.proj_out = zero_module(nn.Linear(channels, channels))

    def forward(self, x):
        h = self.norm(x)
        h = self.attn(h)
        h = h + self.mlp(self.norm(x))
        return x + self.proj_out(h)


class ControlNetClassConditioning(nn.Module):
    """
    ControlNet for class-conditioned control in diffusion models.
    Enables fine-grained class guidance through learnable control modules.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=1000,
        class_dropout_prob=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        lin_layer=nn.Linear,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.class_dropout_prob = class_dropout_prob

        patch_size = patch_size
        num_patches = (input_size // patch_size) ** 2
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        self.class_emb = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.t_emb = TimestepEmbedder(hidden_size)

        self.input_proj = lin_layer(in_channels, hidden_size)

        self.control_blocks = nn.ModuleList(
            [
                ControlNetBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    lin_layer=lin_layer,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj_to_patch = lin_layer(hidden_size, in_channels * patch_size * patch_size)

    def _pos_embed(self, x):
        return x + self.pos_embed

    def forward(self, x, t, y):
        """
        Forward pass for ControlNet.

        Args:
            x: Input features (batch, channels, height, width)
            t: Timestep conditioning (batch,)
            y: Class labels (batch,)

        Returns:
            Control embeddings for each layer
        """
        batch_size = x.shape[0]

        if len(x.shape) == 4:
            x = x.reshape(batch_size, -1, self.hidden_size)

        x = self._pos_embed(x)

        t_emb = self.t_emb(t)
        class_emb = self.class_emb(y)
        c = t_emb + class_emb

        control_outputs = []

        for block in self.control_blocks:
            x = block(x)
            control_outputs.append(x.clone())

        x = self.norm(x)

        return control_outputs, c


class ControlNetAdapter(nn.Module):
    """
    Adapter module that injects ControlNet conditioning into base diffusion model.
    Uses zero-convolution for smooth integration with pre-trained models.
    """

    def __init__(
        self,
        base_model,
        num_classes=1000,
        hidden_size=384,
        control_depth=6,
        control_weight=1.0,
        num_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.base_model = base_model
        self.control_weight = control_weight

        self.control_net = ControlNetClassConditioning(
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=hidden_size,
            depth=control_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
        )

        self.injection_layers = nn.ModuleDict()
        for i in range(control_depth):
            self.injection_layers[f"inject_{i}"] = zero_module(
                nn.Linear(hidden_size, hidden_size)
            )

    def forward(self, x, t, y, return_control=False):
        """
        Forward pass with control injection.

        Args:
            x: Input tensor
            t: Timestep
            y: Class label
            return_control: Whether to return control embeddings

        Returns:
            Output tensor, optionally with control embeddings
        """
        control_outputs, control_cond = self.control_net(x, t, y)

        output = self.base_model(x, t, y)

        if return_control:
            return output, control_outputs, control_cond

        return output


class ControlNetTrainer:
    """Helper class for training and fine-tuning ControlNet adapters."""

    @staticmethod
    def freeze_base_model(model):
        """Freeze all parameters except ControlNet."""
        for name, param in model.named_parameters():
            if "control_net" not in name and "injection" not in name:
                param.requires_grad = False

    @staticmethod
    def get_trainable_params(model):
        """Get only trainable ControlNet parameters."""
        trainable = []
        for name, param in model.named_parameters():
            if ("control_net" in name or "injection" in name) and param.requires_grad:
                trainable.append(param)
        return trainable

    @staticmethod
    def create_optimizer(model, lr=1e-4, weight_decay=0.0):
        """Create optimizer for ControlNet training."""
        trainable_params = ControlNetTrainer.get_trainable_params(model)
        return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
