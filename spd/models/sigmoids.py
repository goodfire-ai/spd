from typing import Any, Literal, override

import torch
from torch import Tensor
from torch.autograd import Function

SigmoidTypes = Literal[
    "leaky_hard",
]


class LowerLeakyHardSigmoidFunction(Function):
    @override
    @staticmethod
    def forward(ctx: Any, x: Tensor, alpha: float = 0.01) -> Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.clamp(x, min=0, max=1)

    @override
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        grad_output = grad_outputs[0]  # Since we only have a single input to the forward method
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha

        # Gradient as if forward pass was alpha * x for x<=0
        grad_input = torch.where(
            x <= 0,
            alpha * grad_output,
            torch.where(x <= 1, grad_output, torch.zeros_like(grad_output)),
        )

        return grad_input, None  # None for alpha gradient since it's not a tensor


def upper_leaky_hard_sigmoid(x: Tensor, alpha: float = 0.01) -> Tensor:
    return torch.where(x > 1, 1 + alpha * (x - 1), torch.clamp(x, min=0, max=1))


def lower_leaky_hard_sigmoid(x: Tensor, alpha: float = 0.001) -> Tensor:
    return LowerLeakyHardSigmoidFunction.apply(x, alpha)  # pyright: ignore[reportReturnType]


SIGMOID_TYPES = {
    "upper_leaky_hard": upper_leaky_hard_sigmoid,
    "lower_leaky_hard": lower_leaky_hard_sigmoid,
}
