from typing import Literal, Optional

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Hand-crafted implementation of Layer Normalization.
    """
    def __init__(self,
                 normalized_shape,
                 eps: float = 0.00001,
                 elementwise_affine: bool = True,
                 bias: bool = True):
        """
        Args:
            normalized_shape (int or list or torch.Size): Input shape from an expected input.
            eps (float): A small value to avoid division by zero.
            elementwise_affine (bool): If True, learnable scale and shift parameters are used.
            bias (bool): If True, adds a learnable bias to the output.
        """
        super(LayerNorm, self).__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            if self.use_bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, layer_mean: Optional[torch.Tensor] = None, layer_var: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for layer normalization.

        Args:
            x (torch.Tensor): Input tensor.
            layer_mean (Optional[torch.Tensor]): Optional precomputed layer mean.
            layer_var (Optional[torch.Tensor]): Optional precomputed layer variance.
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        
        # Determine the dimensions to normalize over (last len(normalized_shape) dims)
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        if layer_mean is None:
            layer_mean = x.mean(dim=dims, keepdim=True)
        if layer_var is None:
            layer_var = (x - layer_mean).pow(2).mean(dim=dims, keepdim=True)

        x_normalized = (x - layer_mean) / torch.sqrt(layer_var + self.eps)

        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight
            if self.bias is not None:
                x_normalized = x_normalized + self.bias

        return x_normalized
    
    def extra_repr(self) -> str:
        return (f'normalized_shape={self.normalized_shape}, eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine}, bias={self.bias}')
    

class LayerNormQuantile(LayerNorm):
    """
    Sparse version of the LayerNorm module.
    This module normalizes only the non-zero elements in the input tensor.
    """
    def __init__(
            self,
            *args,
            sparsity_level: Optional[float] = None,
            quantit_search_mode: Literal['global', 'batchwise', 'channelwise'] = 'channelwise',
            running_stats: bool = False,
            running_shape: Optional[torch.Size] = None,
            momentum: float = 0.1,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        assert sparsity_level is None or (0.0 < sparsity_level < 1.0), \
            "sparsity_level must be in the range (0.0, 1.0)"

        self.sparsity_level = sparsity_level
        self.quantit_search_mode = quantit_search_mode
        self.running_stats = running_stats
        self.momentum = momentum

        self.quantile_view_fn = {
            'global': lambda x: x.view(-1),
            'batchwise': lambda x: x.view(x.size(0), -1),
            'channelwise': lambda x: x.view(x.size(0), x.size(1), -1),
        }[self.quantit_search_mode]

        if self.running_stats:
            self.register_buffer('running_layer_mean', torch.zeros(running_shape or (1, 256, 3)))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def _layer_mean_review(self, layer_mean: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        return layer_mean.view(1, *layer_mean.shape, *((1,) * (len(target_shape) - len(layer_mean.shape) - 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sparse batch normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        layer_mean, layer_var = None, None

        if self.sparsity_level is not None:
            if self.training or self.running_layer_mean is None:
                # Compute quantile threshold
                x_viewed = self.quantile_view_fn(x)
                layer_mean = torch.quantile(x_viewed, self.sparsity_level, dim=-1)
                layer_mean = layer_mean.mean(dim=0) # Average over batch
                layer_mean = self._layer_mean_review(layer_mean, x.shape)

                if self.running_stats:
                    with torch.no_grad():
                        self.running_layer_mean = (1 - self.momentum) * self.running_layer_mean + self.momentum * layer_mean
                        self.num_batches_tracked += 1
            # Compute quantile threshold
            else:
                running_layer_mean = self.running_layer_mean.mean(dim=-1).squeeze(0)
                layer_mean = self._layer_mean_review(running_layer_mean, x.shape)
                
            
        return super().forward(x, layer_mean=layer_mean, layer_var=layer_var)
    
    def extra_repr(self):
        return f'(mean=quantile, var=standard), quantile={self.sparsity_level}, {super().extra_repr()}'

NORMALIZATIONS = {
    'LayerNorm':                 lambda hidden_size: LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
    'LayerNorm-MeanQuantile-50': lambda hidden_size: LayerNormQuantile(hidden_size, sparsity_level=0.5, running_stats=True, elementwise_affine=False, eps=1e-6),
}
