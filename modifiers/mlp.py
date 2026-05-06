import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.common_layers import Mlp

class PolarMlp(Mlp):
    """
    Selective MLP using a lightweight router to simulate neuron sparsity (Polar Sparsity).
    During training, computes routing loss based on absolute magnitude of dense activations.
    During inference, zeros out non-top-k neurons to simulate skipped computation.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., lin_layer=nn.Linear, top_k_ratio=0.5, **kwargs):
        super().__init__(in_features=in_features, hidden_features=hidden_features, out_features=out_features, act_layer=act_layer, drop=drop, lin_layer=lin_layer)
        
        self.top_k_ratio = top_k_ratio
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.num_active_neurons = max(1, int(hidden_features * top_k_ratio))
        
        # Lightweight 2-layer router with a bottleneck
        bottleneck_features = in_features // 4
        self.router = nn.Sequential(
            nn.Linear(in_features, bottleneck_features),
            nn.ReLU(),
            nn.Linear(bottleneck_features, hidden_features)
        )
        self.router_loss = 0.0

    def forward(self, x):
        # 1. Router prediction (token-wise)
        # x is usually (B, N, C)
        router_logits = self.router(x)
        
        # 2. Dense FC1 and Activation
        x1 = self.fc1(x)
        a = self.act(x1)
        
        if self.training:
            # Training: compute exact magnitude per neuron to generate ground truth targets
            magnitudes = torch.abs(a) # (B, N, hidden_features)
            
            # Identify top-k neurons per token
            _, top_k_indices = torch.topk(magnitudes, self.num_active_neurons, dim=-1)
            targets = torch.zeros_like(router_logits)
            targets.scatter_(-1, top_k_indices, 1.0)
            
            # Compute Binary Cross Entropy Loss
            self.router_loss = F.binary_cross_entropy_with_logits(router_logits, targets)
        else:
            self.router_loss = 0.0
            
        # Inference mask based on router
        _, top_k_indices = torch.topk(router_logits, self.num_active_neurons, dim=-1)
        mask = torch.zeros_like(router_logits)
        mask.scatter_(-1, top_k_indices, 1.0)
        
        # Apply mask to output heads
        a = a * mask
        a = self.drop(a)
        
        # Flatten and project
        x_out = self.fc2(a)
        x_out = self.drop(x_out)
        return x_out

MLPS = {
    "Mlp": Mlp,

    "PolarMlp": PolarMlp,
    "PolarMlp_10": partial(PolarMlp, top_k_ratio=0.1),
    "PolarMlp_25": partial(PolarMlp, top_k_ratio=0.25),
    "PolarMlp_50": partial(PolarMlp, top_k_ratio=0.5),
    "PolarMlp_75": partial(PolarMlp, top_k_ratio=0.75),
    "PolarMlp_90": partial(PolarMlp, top_k_ratio=0.9),
}
