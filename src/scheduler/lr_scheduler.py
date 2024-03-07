import numpy as np


class LambdaWarmupCosineScheduler:
    def __init__(
        self,
        warmup_steps,
        lr_min,
        lr_max,
        lr_start,
        max_decay_steps,
        verbosity_interval=0,
    ):
        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_start = lr_start
        self.max_decay_steps = max_decay_steps
        self.verbosity_interval = verbosity_interval
        self.last_lr = 0.0

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.warmup_steps:
            lr = (self.lr_max - self.lr_start) / self.warmup_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.warmup_steps) / (self.max_decay_steps - self.warmup_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + np.cos(np.pi * t)
            )
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)
