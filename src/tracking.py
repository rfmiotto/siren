from typing import Protocol


class NetworkTracker(Protocol):
    def add_batch_metric(self, name: str, value: float, step: int):
        """Implements logging a batch-level metric"""

    def add_epoch_metric(self, name: str, value: float, step: int):
        """Implements logging an epoch-level metric"""
