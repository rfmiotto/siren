import torch


def gradient(target, coords):
    """Compute the gradient with respect to input.

    Parameters
    ----------
    target : torch.Tensor
        2D tensor of shape `(n_coords, ?)` representing the targets.

    coords : torch.Tensor
        2D tensor fo shape `(n_coords, 2)` representing the coordinates.

    Returns
    -------
    grad : torch.Tensor
        2D tensor of shape `(n_coords, 2)` representing the gradient.
    """
    (grad,) = torch.autograd.grad(
        target, coords, grad_outputs=torch.ones_like(target), create_graph=True
    )
    return grad


def divergence(grad, coords):
    """Compute divergence.

    Parameters
    ----------
    grad : torch.Tensor
        2D tensor of shape `(n_coords, 2)` representing the gradient wrt
        x and y.

    coords : torch.Tensor
        2D tensor of shape `(n_coords, 2)` representing the coordinates.

    Returns
    -------
    div : torch.Tensor
        2D tensor of shape `(n_coords, 1)` representing the divergence.

    Notes
    -----
    In a 2D case this will give us f_{xx} + f_{yy}.
    """
    div = 0.0
    for i in range(coords.shape[1]):
        div += gradient(grad[..., i], coords)[..., i : i + 1]

    return div


def laplace(target, coords):
    """Compute laplace operator.

    Parameters
    ----------
    target : torch.Tensor
        2D tesnor of shape `(n_coords, 1)` representing the targets.

    coords : torch.Tensor
        2D tensor of shape `(n_coords, 2)` representing the coordinates.

    Returns
    -------
    torch.Tensor
        2D tensor of shape `(n_coords, 1)` representing the laplace.
    """
    grad = gradient(target, coords)
    return divergence(grad, coords)
