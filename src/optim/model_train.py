import os
import time

import pytorch_lightning as pl
import torch.cuda
from loguru import logger
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_info


class SetUpCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            logger.info("Keyboard interrupt received. Saving model...")
            trainer.save_checkpoint(f"{self.ckptdir}/last.ckpt")

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if (
                    "metrics_over_trainstep_checkpoint"
                    in self.lightning_config["callbacks"]
                ):
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoint"),
                        exist_ok=True,
                    )
            logger.info(f"Project config: \n {OmegaConf.to_yaml(self.config)}")
            OmegaConf.save(
                self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml")
            )
            logger.info(
                f"Lightning config: \n {OmegaConf.to_yaml(self.lightning_config)}"
            )
            OmegaConf.save(
                self.lightning_config,
                os.path.join(self.cfgdir, f"{self.now}_lightning.yaml"),
            )

        else:
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        torch.cuda.reset_peak_memory_stats(self.root_gpu(trainer))
        torch.cuda.synchronize(self.root_gpu(trainer))
        self.start_time = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        torch.cuda.synchronize(self.root_gpu(trainer))
        max_memory = torch.cuda.max_memory_allocated(self.root_gpu(trainer)) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Max memory: {max_memory:.2f} MB")
            rank_zero_info(f"Epoch time: {epoch_time:.2f} s")
        except AttributeError:
            pass

    def root_gpu(self, trainer: pl.Trainer):
        return trainer.strategy.root_device.index
