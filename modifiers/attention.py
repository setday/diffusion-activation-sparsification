import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.common_layers import Attention, _HAS_FLASH_ATTN, resolve_self_attn_mask, maybe_add_mask

if _HAS_FLASH_ATTN:
    from flash_attn import flash_attn_qkvpacked_func


class PolarAttention(Attention):
    """
    Selective Head Attention using a lightweight router to simulate head sparsity.
    During training, computes routing loss based on L2 norms of full head outputs.
    During inference, zeros out non-top-k heads before projection to simulate skipped computation.
    """
    def __init__(self, dim, num_heads=8, top_k_ratio=0.5, **kwargs):
        super().__init__(dim=dim, num_heads=num_heads, **kwargs)
        self.top_k_ratio = top_k_ratio
        self.num_active_heads = max(1, int(num_heads * top_k_ratio))
        
        # Lightweight router
        self.router = nn.Linear(dim, num_heads)
        self.router_loss = 0.0

    def forward(
        self,
        x: torch.Tensor,
        attn_mask=None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        # 1. Router prediction (based on pooled input sequence)
        pooled_x = x.mean(dim=1) # (B, C)
        router_logits = self.router(pooled_x) # (B, num_heads)
        
        # 2. Compute Attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        if _HAS_FLASH_ATTN and x.dtype in [torch.float16, torch.bfloat16] and x.device.type == 'cuda':
            assert isinstance(self.q_norm, nn.Module) and isinstance(self.k_norm, nn.Module)
            assert attn_mask is None
            
            attn_out = flash_attn_qkvpacked_func(
                qkv, 
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=is_causal
            )
            # attn_out is (B, N, num_heads, head_dim)
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            if self.fused_attn:
                attn_out = F.scaled_dot_product_attention(
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
                attn_out = attn @ v

            attn_out = attn_out.transpose(1, 2) # (B, N, num_heads, head_dim)
            
        if self.training:
            # Training: compute exact L2 norm per head to generate ground truth targets
            head_norms = torch.norm(attn_out, p=2, dim=-1).mean(dim=1) # Average norm over seq length -> (B, num_heads)
            
            # Identify top-k heads per batch instance
            _, top_k_indices = torch.topk(head_norms, self.num_active_heads, dim=-1)
            targets = torch.zeros_like(router_logits)
            targets.scatter_(1, top_k_indices, 1.0)
            
            # Compute Binary Cross Entropy Loss
            self.router_loss = F.binary_cross_entropy_with_logits(router_logits, targets)
            
        else:
            self.router_loss = 0.0
            
        # Inference mask based on router
        _, top_k_indices = torch.topk(router_logits, self.num_active_heads, dim=-1)
        mask = torch.zeros_like(router_logits)
        mask.scatter_(1, top_k_indices, 1.0) # (B, num_heads)
        
        # Apply mask to output heads
        mask = mask.unsqueeze(1).unsqueeze(-1) # (B, 1, num_heads, 1)
        attn_out = attn_out * mask

        # Flatten and project
        x = attn_out.reshape(B, N, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


ATTENTIONS = {
    "Attention": Attention,
    
    "PolarAttention": PolarAttention,
    "PolarAttention_10": partial(PolarAttention, top_k_ratio=0.10),
    "PolarAttention_25": partial(PolarAttention, top_k_ratio=0.25),
    "PolarAttention_50": partial(PolarAttention, top_k_ratio=0.5),
    "PolarAttention_75": partial(PolarAttention, top_k_ratio=0.75),
    "PolarAttention_90": partial(PolarAttention, top_k_ratio=0.90),
}
