# DiffiT: Diffusion Vision Transformers for Image Generation
# Based on: https://arxiv.org/abs/2312.02139
#
# This implementation targets LATENT diffusion (with VAE features),
# compatible with the DiT training and sampling pipeline.
#
# Key difference from DiT: uses Time-dependent Multihead Self-Attention (TMSA)
# which integrates temporal conditioning directly into Q/K/V projections,
# instead of adaLN-Zero conditioning.

from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.models.layers import resolve_self_attn_mask, maybe_add_mask
from timm.models.vision_transformer import PatchEmbed

try:
    from flash_attn import flash_attn_qkvpacked_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

from models.common_layers import (
    Attention,
    TimestepEmbedder,
    LabelEmbedder,
    FinalLayer,
    get_2d_sincos_pos_embed,
)


#################################################################################
#                          Core DiffiT Components                               #
#################################################################################

class TMSA(nn.Module):
    """
    Time-dependent Multihead Self-Attention (DiffiT paper, Section 3.1).

    Key differences from standard attention:
    1. Temporal Q/K/V are derived from the conditioning signal and added
       to spatial Q/K/V before computing attention.
    2. A learned relative position bias w^K is applied to attention logits.
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            seq_len: int = 1024,
            attn_head_dim: Optional[int] = None,
            dim_out: Optional[int] = None,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
            lin_layer = nn.Linear,
            device=None,
            dtype=None,
        ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        dim_out = dim_out or dim
        head_dim = attn_head_dim
        if head_dim is None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            head_dim = dim // num_heads
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = Attention._use_fused_attn()
        
        self.seq_len = seq_len

        # Spatial Q/K/V projections
        self.Ws = lin_layer(dim, self.attn_dim * 3, bias=qkv_bias, **dd)

        # Temporal Q/K/V projections
        self.Wt = lin_layer(dim, self.attn_dim * 3, bias=False, **dd)

        # QK-Norm: normalize Q and K per head to prevent attention logit overflow
        # (essential for fp16/bf16 training stability in deep transformers)
        self.q_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()

        # Learned relative position bias w^K (Eq. 5 in the paper):
        # maps each query vector to position-dependent attention bias
        self.WK = lin_layer(self.head_dim, seq_len, bias=False)

        # Output projection
        self.wo = lin_layer(dim, dim, bias=proj_bias, **dd)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(self.attn_dim, **dd) if scale_norm else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        xs: torch.Tensor,
        xt: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            xs: (B, S, C) spatial token features
            xt: (B, C) temporal conditioning vector (t_emb + y_emb)
            attn_mask: optional attention mask (e.g. for causal attention)
            is_causal: whether to apply causal masking (for autoregressive decoding)
        Returns:
            (B, S, C) attention output
        """
        B, N, C = xs.shape

        # Spatial Q/K/V: (B, S, C)
        qkvs = self.qkv(xs)

        # Temporal Q/K/V: (B, C) -> (B, 1, C), broadcast across positions
        qkvt = self.Wt(xt).reshape(B, 1, self.attn_dim * 3)
        
        # Combine spatial + temporal (Eq. 4 in the paper)
        qkv = (qkvs + qkvt).reshape(B, N, 3, self.num_heads, self.head_dim)

        if _HAS_FLASH_ATTN and x.dtype in [torch.float16, torch.bfloat16] and x.device.type == 'cuda':
            # qkv shape requirement for flash_attn_qkvpacked_func: (B, N, 3, H, D)
            # FIXME: Add this to selector
            assert isinstance(self.q_norm, nn.Module) and isinstance(self.k_norm, nn.Module), "Flash Attention requires qk_norm to be True with a valid norm_layer"
            assert attn_mask is None, "Flash Attention does not currently support attention masks"

            pos_bias_k = self.WK.weight.reshape(1, self.seq_len, 1, 1, self.head_dim)
            pos_bias_qv = torch.zeros_like(pos_bias_k)
            pos_bias_qkv = torch.cat([pos_bias_qv, pos_bias_k, pos_bias_qv], dim=2) # (1, seq_len, 3, 1, head_dim)

            qkv = qkv + pos_bias_qkv

            x = flash_attn_qkvpacked_func(
                qkv, 
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=is_causal
            )
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            
            k = k + self.WK.weight # adding pos_bias directly to k is mathematically equivalent to attn + self.WK(q) and more efficient

            # Fall back to PyTorch's native scaled dot product attention
            # (Which also uses FlashAttention underneath if conditions are right on PyTorch 2.0+)
            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    scale=self.scale,
                    is_causal=is_causal
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
                attn = maybe_add_mask(attn, attn_bias)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2)

        x = x.reshape(B, N, self.attn_dim)
        x = self.norm(x)
        x = self.wo(x)
        x = self.proj_drop(x)
        return x


class DiffiTBlock(nn.Module):
    """
    A DiffiT transformer block with TMSA and MLP.

    Unlike DiT which uses adaLN-Zero conditioning, DiffiT integrates
    temporal information directly into the attention mechanism via TMSA.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, seq_len=256,
                 act_layer=lambda: nn.GELU(approximate="tanh"),
                 norm_layer=lambda hidden_size: nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
                 lin_layer=nn.Linear):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.tmsa = TMSA(hidden_size, num_heads, seq_len, lin_layer=lin_layer)
        self.norm2 = norm_layer(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            lin_layer(hidden_size, mlp_hidden_dim),
            act_layer(),
            lin_layer(mlp_hidden_dim, hidden_size),
        )

    def forward(self, x, c):
        """
        Args:
            x: (B, S, C) token features
            c: (B, C) conditioning vector (t_emb + y_emb)
        """
        norm1_out = self.norm1(x)
        tmsa_out = self.tmsa(norm1_out, c)
        x = x + tmsa_out

        norm2_out = self.norm2(x)
        mlp_out = self.mlp(norm2_out)
        x = x + mlp_out

        return x


###############################################################################

class DiffiTResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int, dropout: float, d_ff: int, img_size: int, label_size: int = None, act_layer=lambda: nn.GELU(approximate="tanh"), norm_layer=lambda hidden_size: nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6), lin_layer=nn.Linear):
        super().__init__()
        self.seq_len = img_size * img_size

        self.conv3x3 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1)
        self.swish = nn.SiLU()
        self.group_norm = nn.GroupNorm(num_groups = in_channels//4, num_channels = in_channels)
        self.diffit_block = DiffiTBlock(out_channels, num_heads, dropout, d_ff, img_size, label_size, act_layer=act_layer, norm_layer=norm_layer, lin_layer=lin_layer)

    def forward(self, xs, t, l=None):
        xs_1 = self.conv3x3(self.swish(self.group_norm(xs)))
        xs = xs + self.diffit_block(xs_1, t, l)

        return xs


# From page 14 of the DiffiT paper
class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, output_padding: int = 1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding
        )

    def forward(self, x):
        return self.conv(x)


class ResBlockGroup(nn.Module):
    def __init__(self, L: int, in_channels: int, out_channels: int, num_heads: int, dropout: float, d_ff: int, img_size: int, label_size: int = None, act_layer=lambda: nn.GELU(approximate="tanh"), norm_layer=lambda hidden_size: nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)):
        super().__init__()
        self.diffit_res_block = nn.ModuleList([
            DiffiTResBlock(in_channels, out_channels, num_heads, dropout, d_ff, img_size, label_size, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(L)
        ])

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, l=None):
        for block in self.diffit_res_block:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, l)
        return x



class DiffiTEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, d_ff: int, img_size: int, label_size: int, L1: int = 4, L2: int = 4, L3: int = 4, act_layer=lambda: nn.GELU(approximate="tanh"), norm_layer=lambda hidden_size: nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)):
        super().__init__()
        d_model_2 = d_model*2

        self.diffit_res_block_group_1 = ResBlockGroup(L1, d_model, d_model, num_heads, dropout, d_ff, img_size=img_size, label_size=label_size, act_layer=act_layer, norm_layer=norm_layer)
        self.downsample_1 = Downsample(in_channels=d_model, out_channels=d_model_2)
        self.diffit_res_block_group_2 = ResBlockGroup(L2, d_model_2, d_model_2, num_heads, dropout, d_ff, img_size=img_size//2, label_size=label_size, act_layer=act_layer, norm_layer=norm_layer)
        self.downsample_2 = Downsample(in_channels=d_model_2, out_channels=d_model_2)
        self.diffit_res_block_group_3 = ResBlockGroup(L3, d_model_2, d_model_2, num_heads, dropout, d_ff, img_size=img_size//4, label_size=label_size, act_layer=act_layer, norm_layer=norm_layer)
        self.downsample_3 = Downsample(in_channels=d_model_2, out_channels=d_model_2)

    def forward(self, x, t, l):
        out_1 = self.downsample_1(self.diffit_res_block_group_1(x, t, l))
        out_2 = self.downsample_2(self.diffit_res_block_group_2(out_1, t, l))
        out_3 = self.downsample_3(self.diffit_res_block_group_3(out_2, t, l))
        return [out_1, out_2, out_3]


class DiffiTDecoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, d_ff: int, img_size: int, label_size: int, L1: int = 4, L2: int = 4, L3: int = 4, act_layer=lambda: nn.GELU(approximate="tanh"), norm_layer=lambda hidden_size: nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)):
        super().__init__()
        d_model_2 = d_model*2

        self.upsample_1 = Upsample(in_channels=d_model_2, out_channels=d_model_2)
        self.diffit_res_block_group_3 = ResBlockGroup(L3, d_model_2, d_model_2, num_heads, dropout, d_ff, img_size=img_size//4, label_size=label_size, act_layer=act_layer, norm_layer=norm_layer)
        self.upsample_2 = Upsample(in_channels=d_model_2, out_channels=d_model_2)
        self.diffit_res_block_group_2 = ResBlockGroup(L2, d_model_2, d_model_2, num_heads, dropout, d_ff, img_size=img_size//2, label_size=label_size, act_layer=act_layer, norm_layer=norm_layer)
        self.upsample_3 = Upsample(in_channels=d_model_2, out_channels=d_model)
        self.diffit_res_block_group_1 = ResBlockGroup(L1, d_model, d_model, num_heads, dropout, d_ff, img_size=img_size, label_size=label_size, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x, t, l, skip_connections):
        out_1 = self.diffit_res_block_group_3(self.upsample_1(x + skip_connections[2]), t, l)
        out_2 = self.diffit_res_block_group_2(self.upsample_2(out_1 + skip_connections[1]), t, l)
        out_3 = self.diffit_res_block_group_1(self.upsample_3(out_2 + skip_connections[0]), t, l)
        return out_3


#################################################################################
#                               DiffiT Model                                    #
#################################################################################


class LatentDiffiT(nn.Module):
    """
    DiffiT for latent image generation.

    Architecture parallels DiT:
        Patch embedding -> Transformer blocks -> Final layer -> Unpatchify

    Key difference: Uses TMSA (Time-dependent Multihead Self-Attention)
    instead of standard attention + adaLN-Zero conditioning.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=30,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,

        act_layer=lambda: nn.GELU(approximate="tanh"),
        norm_layer=lambda hidden_size: nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
        lin_layer=nn.Linear,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # Embeddings (same infrastructure as DiT)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Fixed sin-cos positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # DiffiT transformer blocks (each block has its own weights)
        seq_len = num_patches
        self.blocks = nn.ModuleList([
            DiffiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, seq_len=seq_len,
                        act_layer=act_layer, norm_layer=norm_layer, lin_layer=lin_layer)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out TMSA position bias and output projection in each block
        # so that each block starts as identity (via residual connections),
        # analogous to DiT's zero-init of adaLN gating:
        for block in self.blocks:
            nn.init.constant_(block.tmsa.WK.weight, 0)
            nn.init.constant_(block.tmsa.wo.weight, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y):
        """
        Forward pass of DiffiT.
        x: (N, C, H, W) tensor of spatial inputs (latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed   # (N, S, D)
        t = self.t_embedder(t)                      # (N, D)
        y = self.y_embedder(y, self.training)       # (N, D)
        c = t + y                                   # (N, D)

        for i, block in enumerate(self.blocks):
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, c, use_reentrant=True)   # (N, S, D)

        x = self.final_layer(x, c)                  # (N, S, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                      # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiffiT, but also batches the unconditional forward pass
        for classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class ImageDiffiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        in_channels=4,
        hidden_size=1152,
        num_heads=16,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=1000,
        learn_sigma=True,

        L1: int = 2,
        L2: int = 2,
        L3: int = 2,
        L4: int = 2,

        act_layer=lambda: nn.GELU(approximate="tanh"),
        norm_layer=lambda hidden_size: nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
    ):
        super().__init__()
        self.d_ff = hidden_size * mlp_ratio
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        self.encoder = DiffiTEncoder(d_model=hidden_size, num_heads=num_heads, dropout=dropout, d_ff=self.d_ff, img_size=input_size, label_size=num_classes, L1=L1, L2=L2, L3=L3, L4=L4, act_layer=act_layer, norm_layer=norm_layer)
        self.decoder = DiffiTDecoder(d_model=hidden_size, num_heads=num_heads, dropout=dropout, d_ff=self.d_ff, img_size=input_size, label_size=num_classes, L1=L1, L2=L2, L3=L3, L4=L4, act_layer=act_layer, norm_layer=norm_layer)
        self.bottleneck = ResBlockGroup(L4, hidden_size * 2, hidden_size * 2, num_heads, dropout, self.d_ff, img_size=input_size, label_size=num_classes, act_layer=act_layer, norm_layer=norm_layer)

        self.tokenizer = nn.Conv2d(self.in_channels, hidden_size, kernel_size=3, padding=1)
        self.head = nn.Sequential(
            nn.GroupNorm(num_groups=hidden_size//4, num_channels=hidden_size),
            nn.Conv2d(in_channels=hidden_size, out_channels=self.out_channels, kernel_size=3, padding=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize tokenizer and head:
        nn.init.xavier_uniform_(self.tokenizer.weight)
        nn.init.constant_(self.tokenizer.bias, 0)
        nn.init.xavier_uniform_(self.head[1].weight)
        nn.init.constant_(self.head[1].bias, 0)
    
    def forward(self, x, t, y):
        skip_connections = self.encoder(x, t, y)
        bottleneck = self.bottleneck(skip_connections[-1], t, y)
        out = self.decoder(bottleneck, t, y, skip_connections)
        return self.head(out)
    
    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                   DiffiT Configs                              #
#################################################################################

def DiffiT_XL_2(**kwargs):
    return LatentDiffiT(depth=30, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiffiT_XL_4(**kwargs):
    return LatentDiffiT(depth=30, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiffiT_XL_8(**kwargs):
    return LatentDiffiT(depth=30, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiffiT_L_2(**kwargs):
    return LatentDiffiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiffiT_L_4(**kwargs):
    return LatentDiffiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiffiT_L_8(**kwargs):
    return LatentDiffiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiffiT_B_2(**kwargs):
    return LatentDiffiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiffiT_B_4(**kwargs):
    return LatentDiffiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiffiT_B_8(**kwargs):
    return LatentDiffiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiffiT_S_2(**kwargs):
    return LatentDiffiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiffiT_S_4(**kwargs):
    return LatentDiffiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiffiT_S_8(**kwargs):
    return LatentDiffiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiffiT_models = {
    'DiffiT-XL/2': DiffiT_XL_2,  'DiffiT-XL/4': DiffiT_XL_4,  'DiffiT-XL/8': DiffiT_XL_8,
    'DiffiT-L/2':  DiffiT_L_2,   'DiffiT-L/4':  DiffiT_L_4,   'DiffiT-L/8':  DiffiT_L_8,
    'DiffiT-B/2':  DiffiT_B_2,   'DiffiT-B/4':  DiffiT_B_4,   'DiffiT-B/8':  DiffiT_B_8,
    'DiffiT-S/2':  DiffiT_S_2,   'DiffiT-S/4':  DiffiT_S_4,   'DiffiT-S/8':  DiffiT_S_8,
}
