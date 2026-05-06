from typing import Literal
import functools

import torch.nn as nn

from modifiers.decorators import analytical_module, topk_sparse_module


##########################################################################
#                           Sparse activations                           #
##########################################################################

@analytical_module
@topk_sparse_module
class TopKSparseLinear(nn.Linear):
    """
    TopKSparseLinear is a variant of the Linear activation function that applies sparsity to the activations by zeroing out the smallest activations based on a specified sparsity level. The sparsity is applied by keeping only the top k% of the activations, where k is determined by the sparsity_level parameter.
    """
    pass


@analytical_module
@topk_sparse_module
class TopKSparseConv2d(nn.Conv2d):
    """
    TopKSparseConv2d is a variant of the Conv2d activation function that applies sparsity to the activations by zeroing out the smallest activations based on a specified sparsity level. The sparsity is applied by keeping only the top k% of the activations, where k is determined by the sparsity_level parameter.
    """
    pass

@analytical_module
@topk_sparse_module
class TopKSparseConv1d(nn.Conv1d):
    """
    TopKSparseConv1d is a variant of the Conv1d activation function that applies sparsity to the activations by zeroing out the smallest activations based on a specified sparsity level. The sparsity is applied by keeping only the top k% of the activations, where k is determined by the sparsity_level parameter.
    """
    pass


##########################################################################
#          Mapping from string names to activation classes               #
##########################################################################

LINEAR_NAMES_MAP = {
    'Linear': nn.Linear,
    'Conv2d': nn.Conv2d,
    'Conv1d': nn.Conv1d,

    'ALinear': analytical_module(nn.Linear),
    'AConv2d': analytical_module(nn.Conv2d),
    'AConv1d': analytical_module(nn.Conv1d),
    
    'TopKSparseLinear': TopKSparseLinear,
    'TopKSparseConv2d': TopKSparseConv2d,
    'TopKSparseConv1d': TopKSparseConv1d,
    
    'Linear_Q50': functools.partial(TopKSparseLinear, sparsity_level=0.5),
}

LinearClass = Literal[
    'Linear', 'Conv2d', 'Conv1d',
    'ALinear', 'AConv2d', 'AConv1d',
    'TopKSparseLinear', 'TopKSparseConv2d', 'TopKSparseConv1d',
]
