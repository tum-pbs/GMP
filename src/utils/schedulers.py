import math

from torch.optim.lr_scheduler import LambdaLR


class CosineWarmupLR(LambdaLR):
    """Linear warmup for ``warmup_epochs`` (in steps), then cosine decay to
    ``min_lr_factor`` over the remaining training.

    Works at **step granularity** (``interval: step`` in Lightning) so the
    warmup ramps smoothly within each epoch rather than jumping.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_epochs : int
        Number of epochs for the linear warmup phase (from 0 → base LR).
    max_epochs : int
        Total number of training epochs.
    steps_per_epoch : int
        Number of optimizer steps per epoch.  Passed automatically by
        ``configure_optimizers`` — no need to set in the config.
    min_lr_factor : float, optional
        Minimum learning-rate as a fraction of the base LR at the end of
        cosine decay.  Default ``0.0`` (decay all the way to zero).
    last_epoch : int, optional
        The index of the last step.  Default ``-1``.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int | None = None,
        max_epochs: int | None = None,
        steps_per_epoch: int = 1,
        warmup_steps: int | None = None,
        max_steps: int | None = None,
        min_lr_factor: float = 0.0,
        last_epoch: int = -1,
    ):
        if steps_per_epoch is not None and not math.isinf(steps_per_epoch) and not math.isnan(steps_per_epoch):
            steps_per_epoch = int(steps_per_epoch)
        else:
            steps_per_epoch = 1

        if max_steps is not None and max_steps > 0 and not math.isinf(max_steps):
            self.total_steps = int(max_steps)
            if warmup_steps is not None and warmup_steps > 0 and not math.isinf(warmup_steps):
                self.warmup_steps = int(warmup_steps)
            elif warmup_epochs is not None and warmup_epochs > 0 and not math.isinf(warmup_epochs):
                self.warmup_steps = int(warmup_epochs * steps_per_epoch)
            else:
                self.warmup_steps = max(1, int(0.05 * self.total_steps))
        else:
            self.warmup_steps = int((warmup_epochs or 0) * steps_per_epoch)
            self.total_steps = int((max_epochs or 1) * steps_per_epoch)

        self.min_lr_factor = float(min_lr_factor)

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                # Linear warmup: 0 → 1
                return step / max(1, self.warmup_steps)
            # Cosine decay: 1 → min_lr_factor
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.min_lr_factor + 0.5 * (1.0 - self.min_lr_factor) * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)

