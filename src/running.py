from typing import Any, Optional, Tuple
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)

from src.tracking import NetworkTracker
from src.datasets import DatasetReturnItems


class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.epoch = 1
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_fn = torch.nn.MSELoss()
        self.psnr_metric = PeakSignalNoiseRatio()
        self.ssim_metric = StructuralSimilarityIndexMeasure()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_autocast = (
            torch.bfloat16 if self.device.type == "cpu" else torch.float16
        )

        # Send to device
        self.model = self.model.to(device=self.device)
        self.psnr_metric = self.psnr_metric.to(device=self.device)
        self.ssim_metric = self.ssim_metric.to(device=self.device)
        self.loss_fn = self.loss_fn.to(device=self.device)

    def _forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)
        return loss, predictions

    def _backward(self, loss) -> None:
        self.optimizer.zero_grad()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # loss.backward()
        # self.optimizer.step()

    def run(self, tracker: NetworkTracker) -> Tuple[float, torch.Tensor]:
        num_batches = len(self.loader)
        progress_bar = tqdm(enumerate(self.loader), total=num_batches, leave=True)

        epoch_loss = 0.0

        if self.optimizer:
            self.model.train()
        else:
            self.model.eval()

        for batch_index, data in progress_bar:
            data: DatasetReturnItems

            inputs = data["coords"]  # .to(device=self.device, dtype=torch.float32)
            targets = (
                data["intensity"].reshape(-1, 1)
                # .to(device=self.device, dtype=torch.float32)
            )

            if self.optimizer:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype_autocast,
                    cache_enabled=True,
                ):
                    loss, predictions = self._forward(inputs, targets)
                self._backward(loss)
            else:
                with torch.no_grad():
                    loss, predictions = self._forward(inputs, targets)

            # psnr = self.psnr_metric.forward(predictions, targets)
            # ssim = self.ssim_metric.forward(predictions, targets)

            # Update tqdm progress bar
            progress_bar.set_description(f"Epoch {self.epoch}")
            progress_bar.set_postfix(
                loss=f"{loss.item():.5f}",
                # psnr=f"{psnr.item():.5f}",
                # ssim=f"{ssim.item():.5f}",
            )

            tracker.add_batch_metric("loss", loss.item(), batch_index)
            # tracker.add_batch_metric("psnr", psnr.item(), batch_index)
            # tracker.add_batch_metric("ssim", ssim.item(), batch_index)

            epoch_loss += loss.item()

        self.epoch += 1
        epoch_loss = epoch_loss / num_batches
        # epoch_psnr = self.psnr_metric.compute()
        # epoch_ssim = self.ssim_metric.compute()
        epoch_psnr = epoch_loss  # FIXME
        epoch_ssim = epoch_loss  # FIXME

        self.psnr_metric.reset()
        self.ssim_metric.reset()

        return epoch_loss, epoch_psnr, epoch_ssim


def run_epoch(runner: Runner, tracker: NetworkTracker) -> Tuple[float, float]:
    train_epoch_loss, train_epoch_psnr, train_epoch_ssim = runner.run(tracker)

    tracker.add_epoch_metric("loss", train_epoch_loss, runner.epoch)
    tracker.add_epoch_metric("psnr", train_epoch_psnr, runner.epoch)
    tracker.add_epoch_metric("ssim", train_epoch_ssim, runner.epoch)

    return train_epoch_loss, train_epoch_psnr, train_epoch_ssim
