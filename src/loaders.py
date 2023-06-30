from typing import Any
from torch.utils.data.dataloader import DataLoader
from src.datasets import PixelDataset


def get_dataloader(
    img,
    batch_size: int,
    num_workers: int,
    transform: Any,
) -> DataLoader:
    dataset = PixelDataset(img)

    dataset.transform = transform

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    return dataloader
