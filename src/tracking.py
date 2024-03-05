from typing import Optional, Protocol

from numpy.typing import NDArray


class NetworkTracker(Protocol):
    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric"""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging an epoch-level metric"""

    def add_image(self, name: str, img: NDArray, step: int, rescale: Optional[bool]):
        """Implements plotting an epoch-level image"""
