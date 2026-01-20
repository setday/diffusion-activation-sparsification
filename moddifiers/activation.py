import torch.nn as nn


ACTIVATIONS = {
    'GeLU': lambda: nn.GELU(approximate="tanh"),
    'ReLU': lambda: nn.ReLU(),
}
