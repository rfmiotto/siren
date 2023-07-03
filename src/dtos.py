from typing import TypedDict, Union
from typing_extensions import Required, NotRequired


from src.my_types import (
    ArrayFloat32NxN,
    ArrayBoolNxN,
    TensorFloatNx1,
    TensorFloatNx2,
    TensorBoolN,
    TensorFloatN,
)


class DatasetReturnItems(TypedDict):
    coords: TensorFloatN
    derivatives: Union[TensorFloatNx1, TensorFloatNx2]
    mask: TensorBoolN


class TrainingData(TypedDict):
    representation: NotRequired[ArrayFloat32NxN]
    laplacian: NotRequired[ArrayFloat32NxN]
    gradient_x: NotRequired[ArrayFloat32NxN]
    gradient_y: NotRequired[ArrayFloat32NxN]
    mask: Required[ArrayBoolNxN]


class RunnerReturnItems(TypedDict):
    epoch_loss: float
    epoch_psnr: float
    epoch_ssim: float
    predictions: TensorFloatN
    grads: Union[TensorFloatNx1, TensorFloatNx2]
    mask: TensorFloatN
