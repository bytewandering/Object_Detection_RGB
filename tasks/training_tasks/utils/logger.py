"""Merge Lightning's separate train/validation submissions at the same step."""

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


class StepWandbLogger(WandbLogger):
    """Emit one W&B history row per training step, without inventing step values.

    The latest row is buffered until the next step or finalize(), allowing epoch
    summaries to join the final training row. This delays live charts one logging
    interval (and through validation for the epoch's final training row).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_metrics = {}
        self._pending_step = None
        self._last_emitted_step = None

    def _flush_pending(self):
        if self._pending_metrics:
            super().log_metrics(self._pending_metrics, step=self._pending_step)
            self._last_emitted_step = self._pending_step
            self._pending_metrics = {}
            self._pending_step = None

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if not metrics:
            return
        if step is None:
            raise ValueError("StepWandbLogger requires an explicit Lightning training step")
        if self._last_emitted_step is not None and step <= self._last_emitted_step:
            raise ValueError("W&B step already emitted; use a new run when resuming an older checkpoint")
        if self._pending_step is not None and step < self._pending_step:
            raise ValueError("Lightning logging step moved backwards")
        if self._pending_step is not None and step > self._pending_step:
            self._flush_pending()
        self._pending_step = step
        self._pending_metrics.update(metrics)

    @rank_zero_only
    def finalize(self, status):
        self._flush_pending()
        super().finalize(status)
