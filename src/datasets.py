from typing import List, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.my_types import (
    ArrayUintNx2,
    ArrayFloat32NxN,
    ArrayBoolNxN,
    TensorBoolN,
    TensorFloatN,
    TensorFloatNx1,
    TensorFloatNx2,
)
from src.dtos import TrainingData, DatasetReturnItems
from src.hyperparameters import args
from src.transformations import Transform
from src.read_images import read_mat


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
                scaling_factor=1e3,
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

        self.coords_normalized = process_coordinates(self.derivatives, device)

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
    derivatives: Union[TensorFloatNx1, TensorFloatNx2], device: torch.device
) -> TensorFloatN:
    size = int(np.sqrt(derivatives.shape[0]))
    if args.derivatives_from == "filters":
        coords_idx = generate_coordinates(size)
    else:
        coords_idx = pixel_coords_to_cartesian(generate_coordinates(size))
    coords_normalized = 2.0 * ((coords_idx / (size - 1)) - 0.5)
    return torch.from_numpy(coords_normalized).to(device=device, dtype=torch.float)


def generate_coordinates(num_side_points: int) -> ArrayUintNx2:
    """Generate regular grid of 2D coordinates on [0, num_side_points] x [0, num_side_points].

    Parameters
    ----------
    num_side_points : int
        Number of points per dimension.

    Returns
    -------
    pixel_coords : np.ndarray
        Array of row and column coordinates (indices) of shape
        `(num_side_points ** 2, 2)`.
    """
    rows, cols = np.meshgrid(
        range(num_side_points), range(num_side_points), indexing="ij"
    )
    pixel_coords = np.stack([rows.ravel(), cols.ravel()], axis=-1)

    return pixel_coords


def pixel_coords_to_cartesian(pixel_coords: ArrayUintNx2) -> ArrayUintNx2:
    """Convert regular grid of 2D coordinates from pixel domain to Cartesian.
    This process shifts the origin center from the top-left image to the bottom
    left.

    Parameters
    ----------
    pixel_coords : np.ndarray
        Array of row and column coordinates (indices) of shape
        `(num_side_points ** 2, 2)`.

    Returns
    -------
    pixel_coords_cart : np.ndarray
        Array of row and column coordinates (indices) of shape
        `(num_side_points ** 2, 2)`.
    """
    num_side_points = int(np.sqrt(pixel_coords.shape[0]))

    x = pixel_coords.reshape(num_side_points, num_side_points, 2)[:, :, 1]
    y = pixel_coords.reshape(num_side_points, num_side_points, 2)[:, :, 0]

    y = np.flip(y)

    return np.stack([x.ravel(), y.ravel()], axis=1)


def read_coordinates_from_matfile(file: str, device: torch.device) -> TensorFloatN:
    def rescale(array):
        vmax = array.max()
        vmin = array.min()
        return (array - vmin) / (vmax - vmin)

    coordinates_mat = read_mat(file)
    coord_x = rescale(coordinates_mat["coord_x"])
    coord_y = rescale(coordinates_mat["coord_y"])

    coordinates = np.stack([coord_x.ravel(), coord_y.ravel()], axis=-1)

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
