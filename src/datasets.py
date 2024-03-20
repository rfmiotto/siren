from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dtos import DatasetReturnItems, TrainingData
from src.my_types import (
    ArrayBoolNxN,
    ArrayFloat32Nx2,
    ArrayFloat32NxN,
    TensorBoolN,
    TensorFloatNx1,
    TensorFloatNx2,
)
from src.transformations import Transform


class DerivativesPixelDataset(Dataset):
    """Dataset yielding coordinates, derivatives and the (integrated) image.

    Parameters
    ----------
    imagery_items : DatasetImageryItems
        Object containing all the images used in the dataset

    transform: torchvision.transforms

    device : torch.device
        Inform the device where training will occur (`cuda` or `cpu`).

    Attributes
    ----------
    coords_normalized: np.ndarray
        Array of shape `(size ** 2, 2)` representing all normalized coordinates
        of the `img`, from -1 to 1.
        coords_normalized[:, 0] represents the vertical axis, from bottom to top.
        coords_normalized[:, 1] represents the horizontal axis, from left to right.
        That is, the origin is at the bottom-left corner of the image.
    """

    def __init__(
        self,
        images: TrainingData,
        transform: Transform,
        device: torch.device,
    ):
        if has_laplacian(images):
            self.derivatives = process_laplacian_image(
                image=images["laplacian"],
                transform=transform,
                device=device,
                scaling_factor=1e0,
            )

        elif has_gradients(images):
            self.derivatives = process_gradient_images(
                images=[images["gradient_x"], images["gradient_y"]],
                transform=transform,
                device=device,
                scaling_factor=1e1,
            )
        else:
            raise ReferenceError(
                "Derivative data is possibly missing."
                "Check if either the Laplacian or all gradients were passed correctly."
            )

        self.coords_normalized = process_coordinates(images["coordinates"], device)

        self.mask = process_mask(images["mask"])

        self.derivatives = self.derivatives[~self.mask]
        self.coords_normalized = self.coords_normalized[~self.mask]

    def __len__(self):
        """It is the image itself"""
        return 1

    def __getitem__(self, idx: int) -> DatasetReturnItems:
        """Get all relevant data for a single coordinate."""

        return DatasetReturnItems(
            coords=self.coords_normalized.requires_grad_(True),
            derivatives=self.derivatives,
            mask=self.mask,
        )


def process_coordinates(
    coordinates: ArrayFloat32Nx2, device: torch.device
) -> TensorFloatNx2:
    return torch.from_numpy(coordinates).to(device=device, dtype=torch.float)


def process_mask(mask: ArrayBoolNxN) -> TensorBoolN:
    mask = mask.ravel()
    return torch.from_numpy(mask)


def process_laplacian_image(
    image: ArrayFloat32NxN,
    transform: Transform,
    scaling_factor: float,
    device: torch.device,
) -> TensorFloatNx1:
    laplacian = (
        torch.from_numpy(image).to(device=device, dtype=torch.float).reshape(-1, 1)
    )
    laplacian = transform(laplacian)
    laplacian *= scaling_factor
    return laplacian


def process_gradient_images(
    images: List[ArrayFloat32NxN],
    transform: Transform,
    scaling_factor: float,
    device: torch.device,
) -> TensorFloatNx2:
    gradients = np.stack([images[0], images[1]], axis=-1)
    gradients = (
        torch.from_numpy(gradients).to(device=device, dtype=torch.float).view(-1, 2)
    )
    gradients = transform(gradients)
    gradients *= scaling_factor
    return gradients


def has_laplacian(images: TrainingData) -> bool:
    return "laplacian" in images


def has_gradients(images: TrainingData) -> bool:
    return "gradient_x" in images and "gradient_y" in images
