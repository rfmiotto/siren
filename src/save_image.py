from typing import Union
import torch

from src.dtos import RunnerReturnItems
from src.my_types import TensorFloatN, TensorFloatNx1, TensorFloatNx2


def save_laplacian_image(tracker, results: RunnerReturnItems, epoch) -> None:
    vis_data = retrieve_full_image(results["mask"], results["predictions"])
    tracker.add_image("preds", vis_data, epoch)

    vis_data = retrieve_full_image(results["mask"], results["derivatives"])
    tracker.add_image("preds/laplacian", vis_data, epoch)


def save_gradient_images(tracker, results: RunnerReturnItems, epoch) -> None:
    gradx, grady = (
        results["derivatives"][..., 0],
        results["derivatives"][..., 1],
    )

    vis_data = retrieve_full_image(results["mask"], results["predictions"])
    tracker.add_image("preds", vis_data, epoch)

    vis_data = retrieve_full_image(results["mask"], gradx.view(1, -1, 1))
    tracker.add_image("preds/gradx", vis_data, epoch)

    vis_data = retrieve_full_image(results["mask"], grady.view(1, -1, 1))
    tracker.add_image("preds/grady", vis_data, epoch)


def should_save_image(epoch: int, epochs_until_summary) -> bool:
    return not epoch % epochs_until_summary


def retrieve_full_image(
    mask: TensorFloatN, variable: Union[TensorFloatN, TensorFloatNx1, TensorFloatNx2]
) -> TensorFloatNx1:
    """
    Retrieve the full image from `mask`. The `variable` to be plotted is a subset
    of the mask (where mask is false). So we retrieve the image with the same
    shape of the mask and place the variable where mask is False. Regions where
    mask is True will be filled with ones.
    """
    full_size_variable = torch.ones_like(mask, dtype=torch.float).view(1, -1, 1)
    # fmt: off
    full_size_variable *= variable.detach().cpu().min() # make masked regions black
    # fmt: on
    full_size_variable[~mask] = variable.detach().cpu()
    return full_size_variable
