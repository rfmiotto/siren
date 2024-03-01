import numpy as np
import torch


def original_siren_initialization(
    weight: torch.Tensor, is_first: bool = False, omega: float = 1
):
    """Initialize the weigth of the Linear layer.

    Parameters
    ----------
    weight : torch.Tensor
        The learnable 2D weight matrix.

    is_first : bool
        If True, this Linear layer is the very first one in the network.

    omega : float
        Hyperparamter.
    """
    in_features = weight.shape[1]

    with torch.no_grad():
        bound = 1 / in_features if is_first else np.sqrt(6 / in_features) / omega
        weight.uniform_(-bound, bound)
