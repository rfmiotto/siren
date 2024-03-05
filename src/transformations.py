from functools import partial
from typing import Callable, Literal, Union

import numpy as np
import torch
from PIL import Image

from src.my_types import TensorFloatNx1, TensorFloatNx2

DerivativeTensor = Union[TensorFloatNx1, TensorFloatNx2]
Transform = Callable[[DerivativeTensor], DerivativeTensor]


def get_transform_fn(
    option: Literal["image", "value", "self", "no_transform"]
) -> Callable:
    """
    The derivative images are in the pixel domain, so their values range from
    0 to 1. We need to specify which color represents the zero derivatives.
    This is a higher order function (HOF) that returns a function that
    shift-scales the values of the images containing the derivatives.

    Parameters
    ----------
    option : "image" or "value"
        If "image", the mean will be computed from the image passed as argument;
        If "value", the mean wil be set to the value (in the range [0, 1]);
        If "self", the mean will be taken from the derivative tensor;
        If "no_transform", no transform will be applied;
    """

    # TODO   for now, it only works for a single value. So, for the case of
    # TODO   gradients, for example, we cannot specify separate values for x
    # TODO   and y components yet.

    match option:
        case "image":
            return shift_scale_img
        case "value":
            return partial(shift_scale_value, mean=0.0)  # TODO user-defined mean
        case "self":
            return shift_scale_self
        case "no_transform":
            return no_transform
        case _:
            raise ReferenceError(
                "Option not found. Choose between 'image', 'value' or 'self'."
            )


def no_transform(tensor: DerivativeTensor) -> DerivativeTensor:
    return tensor


def shift_scale_img(tensor: DerivativeTensor) -> DerivativeTensor:
    std = torch.std(tensor)
    image = Image.open("reset_laplacian.png")
    mean = np.float32(np.average(image) / 255.0)
    # return (tensor - mean) / std
    return tensor - mean  # FIXME deixa ou nao o std???


def shift_scale_self(tensor: DerivativeTensor) -> DerivativeTensor:
    std, mean = torch.std_mean(tensor)
    return (tensor - mean) / std


def shift_scale_value(tensor: DerivativeTensor, mean: float) -> DerivativeTensor:
    std = torch.std(tensor)
    return (tensor - mean) / std
