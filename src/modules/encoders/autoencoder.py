import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from modules.diffusionmodules.diagonalgaussian import DiagonalGaussian
from modules.diffusionmodules.models import Decoder, Encoder
from modules.utils import instantiate_from_config


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        config,
        lossconfig,
        emb_dim,
        ckpt_path,
        tile_size=32,
        overlap_ratio=0.25,
        ignore_keys=None,
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        use_tiling=False,
        **kwargs,
    ):
        super().__init__()
        if ignore_keys is None:
            ignore_keys = []
        self.image_key = image_key
        self.encoder = Encoder(**config)
        self.decoder = Decoder(**config)
        self.loss = instantiate_from_config(lossconfig)
        assert config["double_z"]
        self.quant_conv = nn.Conv2d(2 * config["z_channels"], 2 * emb_dim, 1)
        self.post_quant_conv = nn.Conv2d(emb_dim, config["z_channels"], 1)
        self.embedding_dim = emb_dim
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.use_tiling = use_tiling
        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_checkpoint(self, ckpt_path, ignore_keys):
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    logger.info(f"Ignoring key {k}")
                    del state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        logger.info(f"Restored model from {ckpt_path}")

    def encode(self, z):
        if self.use_tiling:
            return self.tiled_encode(z)
        h = self.encoder(z)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussian(moments)
        return posterior

    def decode(self, z):
        if self.use_tiling:
            return self.tiled_decode(z)
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def blend_h(self, row, tile, blend_extend):
        blend_extend = min(row.shape[3], tile.shape[3], blend_extend)
        for x in range(blend_extend):
            tile[:, :, :, x] = row[:, :, :, -blend_extend + x] * (
                1 - x / blend_extend
            ) + tile[:, :, :, x] * (x / blend_extend)
        return tile

    def blend_v(self, row, tile, blend_extend):
        blend_extend = min(row.shape[2], tile.shape[2], blend_extend)
        for y in range(blend_extend):
            tile[:, :, y, :] = row[:, :, -blend_extend + y, :] * (
                1 - y / blend_extend
            ) + tile[:, :, y, :] * (y / blend_extend)
        return tile

    def tiled_encode(self, z):
        overlap_size = int(self.tile_size * (1 - self.overlap_ratio))
        blend_extend = int(self.tile_size * self.overlap_ratio)
        row_limit = self.tile_size - blend_extend

        rows = list()
        for i in range(0, z.shape[2], overlap_size):
            row = list()
            for j in range(0, z.shape[3], overlap_size):
                z_tile = z[:, :, i : i + self.tile_size, j : j + self.tile_size]
                tile = self.encode(z_tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = list()
        for i, row in enumerate(rows):
            result_row = list()
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extend)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extend)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        posterior = DiagonalGaussian(moments)
        return posterior

    def tiled_decode(self, z):
        overlap_size = int(self.tile_size * (1 - self.overlap_ratio))
        blend_extend = int(self.tile_size * self.overlap_ratio)
        row_limit = self.tile_size - blend_extend

        rows = list()
        for i in range(0, z.shape[2], overlap_size):
            row = list()
            for j in range(0, z.shape[3], overlap_size):
                z_tile = z[:, :, i : i + self.tile_size, j : j + self.tile_size]
                self.post_quant_conv(z_tile)
                tile = self.decode(z_tile)
                row.append(tile)
            rows.append(row)
        result_rows = list()
        for i, row in enumerate(rows):
            result_row = list()
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extend)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extend)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        return dec

    def forward(self, x, sample_posterior=True):
        posterior = self.encoder(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decoder(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        recon, posterior = self(inputs)

        if optimizer_idx == 0:
            ae_loss, log_dict_ae = self.loss(
                inputs,
                recon,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log(
                "ae_loss",
                ae_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return ae_loss

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(
                inputs,
                recon,
                posterior,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log(
                "discloss",
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        recon, posterior = self(inputs)
        ae_loss, log_dict_ae = self.loss(
            inputs,
            recon,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        disc_loss, log_dict_disc = self.loss(
            inputs,
            recon,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self) -> OptimizerLRScheduler:
        lr = self.learning_rate
        optimizer_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        optimizer_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [optimizer_ae, optimizer_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            x_recon, posterior = self(x)
            if x.shape[1] > 3:
                assert x.shape[1] > 3
                x = self.to_rgb(x)
                x_recon = self.to_rgb(x_recon)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["recon"] = x_recon
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
