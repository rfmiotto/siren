import torch

from src.my_types import TensorFloatNx1, TensorFloatNx2, TensorScalar
from src.vector_ops import gradient, laplace

loss_fn = torch.nn.MSELoss()


class FitGradients(torch.nn.Module):
    def forward(
        self,
        y_pred: TensorFloatNx2,
        grad_y_true: TensorFloatNx2,
        coords: TensorFloatNx2,
    ) -> TensorScalar:
        grads = gradient(y_pred, coords)
        return loss_fn(grads, grad_y_true), grads


class FitLaplacian(torch.nn.Module):
    def forward(
        self,
        y_pred: TensorFloatNx1,
        lapl_y_true: TensorFloatNx1,
        coords: TensorFloatNx2,
    ) -> TensorScalar:
        lapl = laplace(y_pred, coords)
        return loss_fn(lapl, lapl_y_true), lapl
