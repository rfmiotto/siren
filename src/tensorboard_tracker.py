import re
import time
from pathlib import Path, PurePath

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.tensorboard import SummaryWriter

from src.my_types import TensorFloatN, TensorFloatNxN
from src.validate_dir import validate_dir


class TensorboardTracker:
    def __init__(self, log_dir: str, filename: str = "", create: bool = True):
        default_name = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        filename = filename if filename else default_name
        validate_dir(log_dir + "/" + filename, create=create)
        self._writer = SummaryWriter(log_dir + "/" + filename)

    def flush(self):
        self._writer.flush()

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f"train/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f"train/epoch/{name}"
        self._writer.add_scalar(tag, value, step)

    @staticmethod
    def _rescale_image(img: TensorFloatNxN) -> TensorFloatN:
        xmax = torch.max(img[:, 2:-2, 2:-2])
        xmin = torch.min(img[:, 2:-2, 2:-2])
        img[img > xmax] = xmax
        img[img < xmin] = xmin
        img_rescaled = (img - xmin) / (xmax - xmin)
        return img_rescaled

    def add_image(self, name: str, img: TensorFloatN, step: int, rescale=True):
        batch_size = img.shape[0]
        image_size = int(np.sqrt(img.shape[1]))
        img_formatted = img.view(batch_size, image_size, image_size)

        subdirectories = re.split("/", name)[:-1]
        filename = re.split("/", name)[-1]
        filename += f"_{step}.mat"
        path = PurePath("outputs", *subdirectories)
        Path(path).mkdir(parents=True, exist_ok=True)
        full_path_outfile = PurePath(path, filename)
        savemat(full_path_outfile, {"data": img_formatted.cpu()})

        if rescale:
            img_formatted = self._rescale_image(img_formatted)
        self._writer.add_image(name, img_formatted, global_step=step)
