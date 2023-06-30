from typing import TypedDict
import numpy as np
import torch
from scipy.ndimage import laplace, sobel
from torch.utils.data import Dataset
from PIL import Image
import skimage
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from src.my_types import ArrayNxN, ArrayNx2, ArrayNxNx1, ArrayNxNx2


def generate_coordinates(
    num_side_points: int,
) -> ArrayNx2[np.int_]:
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


class DatasetReturnItems(TypedDict):
    coords_idx: ArrayNx2
    coords: ArrayNx2
    intensity: ArrayNxNx1
    grad: ArrayNxNx2
    grad_norm: ArrayNxNx1
    laplace: ArrayNxNx1


class PixelDataset(Dataset):
    """Dataset yielding coordinates, intensitives and (higher) derivatives.

    Parameters
    ----------
    img : np.ndarray
        2D image representing a grayscale image.

    Attributes
    ----------
    size : int
        Height and width of the square image.

    coords_idx : np.ndarray
        Array of shape `(size ** 2, 2)` representing all coordinates of the
        `img`.

    grad : np.ndarray
        Array of shape `(size, size, 2)` representing the approximate
        gradient in the two directions.

    grad_norm : np.ndarray
        Array of shape `(size, size)` representing the approximate gradient
        norm of `img`.

    laplace : np.ndarray
        Array of shape `(size, size)` representing the approximate laplace operator.
    """

    def __init__(self, img: ArrayNxN):
        is_square_image = img.shape[0] == img.shape[1]
        is_2d_image = img.ndim == 2

        if not (is_2d_image and is_square_image):
            raise ValueError("Only 2D square images are supported.")

        self.transform = Compose(
            [
                Resize(270),  # FIXME
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
            ]
        )

        # self.img = img  # FIXME
        img = skimage.data.camera()
        self.size = 270  # img.shape[0] # FIXME
        self.img = self.transform(Image.fromarray(img)).view(self.size, self.size)
        self.coords_idx = generate_coordinates(self.size)
        self.grad = np.stack(
            [sobel(self.img, axis=0), sobel(self.img, axis=1)], axis=-1
        )
        self.grad_norm = np.linalg.norm(self.grad, axis=-1)
        self.laplace = laplace(self.img)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.img = torch.from_numpy(self.img).to(device=device)
        self.img = self.img.to(device=device)
        self.size = torch.tensor(self.size)
        self.coords_idx = torch.from_numpy(self.coords_idx).to(device=device)
        self.grad = torch.from_numpy(self.grad).to(device=device)
        self.grad_norm = torch.from_numpy(self.grad_norm).to(device=device)
        self.laplace = torch.from_numpy(self.laplace).to(device=device)

    # def __len__(self):
    #     """Determine the number of samples (pixels)."""
    #     return self.size**2

    # def __getitem__(self, idx):
    #     """Get all relevant data for a single coordinate."""
    #     coords_idx = self.coords_idx[idx]

    #     coords_normalized = 2.0 * ((coords_idx / (self.size - 1)) - 0.5)

    #     row, col = coords_idx

    #     return DatasetReturnItems(
    #         coords=coords_normalized,
    #         coords_idx=coords_idx,
    #         intensity=self.img[row, col],
    #         grad_norm=self.grad_norm[row, col],
    #         grad=self.grad[row, col],
    #         laplace=self.laplace[row, col],
    #     )

    # return DatasetReturnItems(
    #     coords=torch.from_numpy(coords_normalized),
    #     coords_idx=torch.from_numpy(coords_idx),
    #     intensity=torch.tensor(self.img[row, col]),
    #     grad_norm=torch.tensor(self.grad_norm[row, col]),
    #     grad=torch.from_numpy(self.grad[row, col]),
    #     laplace=torch.tensor(self.laplace[row, col]),
    # )

    def __len__(self):
        """It is the image itself"""
        return 1

    def __getitem__(self, idx):
        """Get all relevant data for a single coordinate."""
        coords_idx = self.coords_idx

        coords_normalized = 2.0 * ((coords_idx / (self.size - 1)) - 0.5)

        return DatasetReturnItems(
            coords=coords_normalized,
            coords_idx=coords_idx,
            intensity=self.img,
            grad_norm=self.grad_norm,
            grad=self.grad,
            laplace=self.laplace,
        )

    # return DatasetReturnItems(
    #     coords=torch.from_numpy(coords_normalized),
    #     coords_idx=torch.from_numpy(coords_idx),
    #     intensity=torch.tensor(self.img[row, col]),
    #     grad_norm=torch.tensor(self.grad_norm[row, col]),
    #     grad=torch.from_numpy(self.grad[row, col]),
    #     laplace=torch.tensor(self.laplace[row, col]),
    # )
