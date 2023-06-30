import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class TensorboardTracker:
    def __init__(self, log_dir: str, filename: str = "", create: bool = True):
        self._validate_log_dir(log_dir, create=create)
        default_name = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        filename = filename if filename else default_name
        self._writer = SummaryWriter(log_dir + "/" + filename)

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        path = Path(log_dir).resolve()
        if path.exists():
            return
        if not path.exists() and create:
            path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def flush(self):
        self._writer.flush()

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f"train/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f"train/epoch/{name}"
        self._writer.add_scalar(tag, value, step)
