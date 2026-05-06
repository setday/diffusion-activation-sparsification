from typing import Literal, Optional

import torch
import torch.nn as nn

from modifiers.utils import review_as_with_batch


##########################################################################
#       Hand-crafted implementations of normalization layers             #
##########################################################################


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

    def forward(
            self,
            x: torch.Tensor,
            layer_mean: Optional[torch.Tensor] = None,
            layer_var: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
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
            quantile_search_mode: Literal['global', 'batchwise', 'channelwise'] = 'channelwise',
            running_stats: bool = False,
            running_shape: Optional[torch.Size] = None,
            momentum: float = 0.1,
            max_tracked_cnt: Optional[int] = 50000,
            **kwargs
        ):
        """
        Args:
            *args: Positional arguments for the base LayerNorm class.
            sparsity_level (Optional[float]): Fraction of elements to consider as non-zero (between
                0.0 and 1.0). If None, behaves like standard LayerNorm.
            quantile_search_mode (str): Method to compute quantile threshold. Options are:
                - 'global': Compute quantile across all elements in the input tensor.
                - 'batchwise': Compute quantile separately for each sample in the batch.
                - 'channelwise': Compute quantile separately for each channel (default).
        """

        super().__init__(*args, **kwargs)

        assert sparsity_level is None or (0.0 < sparsity_level < 1.0), \
            "sparsity_level must be in the range (0.0, 1.0)"

        self.sparsity_level = sparsity_level
        self.quantile_search_mode = quantile_search_mode
        self.running_stats = running_stats
        self.momentum = momentum

        self.quantile_view_fn = {
            'global': lambda x: x.view(-1),
            'batchwise': lambda x: x.view(x.size(0), -1),
            'channelwise': lambda x: x.view(x.size(0), x.size(1), -1),
        }[self.quantile_search_mode]

        if self.running_stats:
            self.register_buffer('running_layer_mean', torch.zeros(running_shape or (384)))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            
        self.max_tracked_cnt = max_tracked_cnt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sparse batch normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        layer_mean, layer_var = None, None

        if self.sparsity_level is None:
            return super().forward(x)

        if self.running_stats and self.max_tracked_cnt is not None and self.max_tracked_cnt <= self.num_batches_tracked:
            if len(self.running_layer_mean.shape) != 1:
                self.running_layer_mean = self.running_layer_mean.mean(dim=(0,-1))
            layer_mean = self.running_layer_mean
        elif self.training or self.running_layer_mean is None:
            # Compute quantile threshold
            x_viewed = self.quantile_view_fn(x)
            # layer_mean = torch.quantile(x_viewed, self.sparsity_level, dim=-1)
            n_remove = round(self.sparsity_level * x_viewed.size(dim=-1))
            layer_mean = torch.kthvalue(x_viewed, n_remove, dim=-1).values
            layer_mean = layer_mean.mean(dim=0) # Average over batch

            if self.running_stats:
                with torch.no_grad():
                    if self.num_batches_tracked != 0:
                        self.running_layer_mean = (1 - self.momentum) * self.running_layer_mean + self.momentum * layer_mean
                    else:
                        self.running_layer_mean = layer_mean
                    self.num_batches_tracked += 1
        # Compute quantile threshold
        else:
            if len(self.running_layer_mean.shape) != 1:
                self.running_layer_mean = self.running_layer_mean.mean(dim=(0,-1))
            layer_mean = self.running_layer_mean

        layer_mean = review_as_with_batch(layer_mean, x.shape)
            
        return super().forward(x, layer_mean=layer_mean, layer_var=layer_var)
    
    def extra_repr(self):
        return f'(mean=quantile, var=standard), quantile={self.sparsity_level}, {super().extra_repr()}'


##########################################################################
#                 List of available normalization layers                 #
##########################################################################


NORMALIZATIONS = {
    'LayerNorm':                 lambda hidden_size: LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),

    'LayerNorm-MeanQuantile-10': lambda hidden_size: LayerNormQuantile(hidden_size, sparsity_level=0.1, elementwise_affine=False, eps=1e-6),
    'LayerNorm-MeanQuantile-25': lambda hidden_size: LayerNormQuantile(hidden_size, sparsity_level=0.25, elementwise_affine=False, eps=1e-6),
    'LayerNorm-MeanQuantile-50': lambda hidden_size: LayerNormQuantile(hidden_size, sparsity_level=0.5, elementwise_affine=False, eps=1e-6),
    'LayerNorm-MeanQuantile-75': lambda hidden_size: LayerNormQuantile(hidden_size, sparsity_level=0.75, elementwise_affine=False, eps=1e-6),
    'LayerNorm-MeanQuantile-90': lambda hidden_size: LayerNormQuantile(hidden_size, sparsity_level=0.9, elementwise_affine=False, eps=1e-6),
}
