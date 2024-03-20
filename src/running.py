from dataclasses import dataclass
from typing import Any, Literal, Tuple, TypedDict, Union

import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.datasets import DatasetReturnItems
from src.dtos import RunnerReturnItems
from src.hyperparameters import args
from src.losses import FitGradients, FitLaplacian
from src.metrics import relative_residual_error
from src.my_types import TensorFloatNx1, TensorFloatNx2
from src.save_image import save_gradient_images, save_laplacian_image, should_save_image
from src.tracking import NetworkTracker


class TrainingConfig(TypedDict):
    fit_option: Literal["gradients, laplacian"]
    optimizer: torch.optim.Optimizer
    scaler: torch.cuda.amp.GradScaler
    device: torch.device
    dtype_autocast: torch.dtype
    use_autocast: bool


@dataclass
class TrainingMetrics:
    psnr_metric = PeakSignalNoiseRatio()
    ssim_metric = StructuralSimilarityIndexMeasure()


class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        config: TrainingConfig,
        metrics: TrainingMetrics,
    ):
        self.epoch = 0
        self.loader = loader
        self.model = model
        self.optimizer = config["optimizer"]
        self.use_autocast = config["use_autocast"]
        self.scaler = config["scaler"]
        is_laplacian = config["fit_option"] == "laplacian"
        self.loss_fn = FitLaplacian() if is_laplacian else FitGradients()
        self.device = config["device"]
        self.dtype_autocast = config["dtype_autocast"]

        # Send to device
        self.model = self.model.to(device=self.device)
        self.psnr_metric = metrics.psnr_metric.to(device=self.device)
        self.ssim_metric = metrics.ssim_metric.to(device=self.device)
        self.loss_fn = self.loss_fn.to(device=self.device)

    def _forward(
        self, inputs: TensorFloatNx2, targets: Union[TensorFloatNx1, TensorFloatNx2]
    ):
        predictions = self.model(inputs)
        loss, grads = self.loss_fn(predictions, targets, inputs)
        return loss, predictions, grads

    def _backward_with_autocast(self, loss) -> None:
        self.optimizer.zero_grad()

        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.scaler.get_scale()
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _backward(self, loss) -> None:
        self.optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        loss.backward()
        self.optimizer.step()

    def _set_train_mode(self) -> None:
        self.model.train()

    def _run_batch(self, inputs, targets):
        if self.use_autocast:
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype_autocast,
                cache_enabled=True,
                enabled=self.use_autocast,
            ):
                loss, predictions, derivatives = self._forward(inputs, targets)
            self._backward_with_autocast(loss)
        else:
            loss, predictions, derivatives = self._forward(inputs, targets)
            self._backward(loss)
        return loss, predictions, derivatives

    def run(self) -> RunnerReturnItems:
        num_batches = len(self.loader)

        epoch_loss = 0.0

        self._set_train_mode()

        # pylint: disable=unused-variable
        for batch_index, data in enumerate(self.loader):
            data: DatasetReturnItems

            inputs = data["coords"]
            targets = data["derivatives"]
            mask = data["mask"]

            loss, predictions, derivatives = self._run_batch(inputs, targets)

            # psnr = self.psnr_metric.forward(inputs, derivatives)
            # ssim = self.ssim_metric.forward(inputs, derivatives)

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / num_batches
        # epoch_psnr = self.psnr_metric.compute()
        # epoch_ssim = self.ssim_metric.compute()
        epoch_psnr = relative_residual_error(predictions, targets)  # FIXME
        epoch_ssim = epoch_loss  # FIXME

        self.psnr_metric.reset()
        self.ssim_metric.reset()

        self.epoch += 1

        return RunnerReturnItems(
            epoch_loss=epoch_loss,
            epoch_psnr=epoch_psnr,
            epoch_ssim=epoch_ssim,
            predictions=predictions,
            derivatives=derivatives,
            mask=mask,
        )


def run_epoch(runner: Runner, tracker: NetworkTracker) -> Tuple[float, float, float]:
    results = runner.run()

    tracker.add_epoch_metric("loss", results["epoch_loss"], runner.epoch)
    tracker.add_epoch_metric("psnr", results["epoch_psnr"], runner.epoch)
    tracker.add_epoch_metric("ssim", results["epoch_ssim"], runner.epoch)

    if should_save_image(runner.epoch, args.epochs_until_summary):
        if args.fit == "gradients":
            save_gradient_images(tracker, results, runner.epoch)
        else:
            save_laplacian_image(tracker, results, runner.epoch)

    return results["epoch_loss"], results["epoch_psnr"], results["epoch_ssim"]
