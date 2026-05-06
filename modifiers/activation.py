from typing import Literal, Optional

import torch
import torch.nn as nn

from modifiers.utils import review_as_with_batch

from modifiers.decorators import analytical_module


@analytical_module
class SparseGELU(nn.GELU):
    def __init__(
        self,
        sparsity_level: Optional[float] = None,
        quantile_search_mode: Literal['global', 'batchwise', 'channelwise'] = 'channelwise',
        running_stats: bool = True,
        running_shape: Optional[torch.Size] = None,
        momentum: float = 0.1,
        max_tracked_cnt: Optional[int] = 50000,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert sparsity_level is None or (0.0 < sparsity_level < 1.0), "sparsity_level must be in (0, 1)"

        self.sparsity_level = sparsity_level
        self.running_stats = running_stats
        self.momentum = momentum

        self.quantile_search_mode = quantile_search_mode
        self.quantile_view_fn = {
            'global': lambda x: x.view(-1),
            'batchwise': lambda x: x.view(x.size(0), -1),
            'channelwise': lambda x: x.view(x.size(0), x.size(1), -1),
        }[self.quantile_search_mode]

        if self.running_stats:
            self.register_buffer('running_treshold', torch.zeros(running_shape or (256,)))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            
        self.max_tracked_cnt = max_tracked_cnt

    def forward(self, x):
        x_act = super().forward(x)

        if self.sparsity_level is None:
            return x_act
        
        if self.running_stats and self.max_tracked_cnt is not None and self.max_tracked_cnt <= self.num_batches_tracked:
            treshold = self.running_treshold
        elif self.training or self.running_treshold is None:
            # Compute quantile threshold
            x_viewed = self.quantile_view_fn(x_act)
            n_remove = round(self.sparsity_level * x_viewed.size(dim=-1))
            treshold = torch.kthvalue(x_viewed, n_remove, dim=-1).values
            treshold = treshold.mean(dim=0) # Average over batch

            if self.running_stats:
                with torch.no_grad():
                    if self.num_batches_tracked != 0:
                        self.running_treshold = (1 - self.momentum) * self.running_treshold + self.momentum * treshold
                    else:
                        self.running_treshold = treshold
                    self.num_batches_tracked += 1
        # Compute quantile threshold
        else:
            treshold = self.running_treshold
        
        treshold = review_as_with_batch(treshold, x_act.shape)
        mask = x_act < treshold
        x_act.masked_fill_(mask, 0.0)

        return x_act


##########################################################################
#                List of available activation functions                 #
##########################################################################


ACTIVATIONS = {
    'ReLU': lambda: nn.ReLU(),
    'SiLU': lambda: nn.SiLU(),
    'GELU': lambda: nn.GELU(approximate="tanh"),
    
    'AReLU': lambda: analytical_module(nn.ReLU)(),
    'AGeLU': lambda: analytical_module(nn.GELU)(approximate="tanh"),
    'ASiLU': lambda: analytical_module(nn.SiLU)(),

    'GELU_Q10': lambda: SparseGELU(sparsity_level=0.1, approximate="tanh"),
    'GELU_Q25': lambda: SparseGELU(sparsity_level=0.25, approximate="tanh"),
    'GELU_Q50': lambda: SparseGELU(sparsity_level=0.5, approximate="tanh"),
    'GELU_Q75': lambda: SparseGELU(sparsity_level=0.75, approximate="tanh"),
    'GELU_Q90': lambda: SparseGELU(sparsity_level=0.9, approximate="tanh"),
}
