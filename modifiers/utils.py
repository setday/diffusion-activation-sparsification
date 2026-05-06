from torch import Tensor

def review_as_with_batch(x: Tensor, target_shape: Tensor) -> Tensor:
    extra_dims = len(target_shape) - len(x.shape) - 1
    return x.view(1, *x.shape, *((1,) * extra_dims))
