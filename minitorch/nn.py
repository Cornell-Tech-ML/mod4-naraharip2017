from typing import Tuple, Optional

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    output = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    output = output.permute(0, 1, 2, 4, 3, 5).contiguous()
    output = output.view(batch, channel, new_height, new_width, kh * kw)
    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to input with kernel size"""
    input_tiled, new_height, new_weight = tile(input, kernel)
    return input_tiled.mean(dim=4).view(
        input_tiled.shape[0], input_tiled.shape[1], new_height, new_weight
    )


# TODO: Implement for Task 4.3.


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Max function applied to tensor elements"""
        max_vals = a.f.max_reduce(a, int(dim.item()))
        ctx.save_for_backward(max_vals, a, dim)
        return max_vals

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Derivative of max function"""
        (max_vals, a, dim) = ctx.saved_values
        arg_max = a.f.eq_zip(a, max_vals)
        return grad_output * arg_max, 0.0


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Apply max reduction to input tensor"""
    if dim is not None:
        return Max.apply(input, input._ensure_tensor(dim))
    else:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor over input dimension"""
    input_exp = input.exp()
    return input_exp / input_exp.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor over input dimension"""
    max_input = max(input, dim)
    input_normalized = input - max_input
    log_sum_exp = input_normalized.exp().sum(dim).log()
    return input_normalized - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to input with kernel size"""
    input_tiled, new_height, new_weight = tile(input, kernel)
    return max(input_tiled, 4).view(
        input_tiled.shape[0], input_tiled.shape[1], new_height, new_weight
    )


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, includes an argument to turn off"""
    if ignore:
        return input

    random_numbers = rand(input.shape)

    mask = random_numbers > p

    return input * mask
