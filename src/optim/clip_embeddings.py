"""
Taken and modified from:
 https://github.com/huggingface/diffusers/blob/12004bf3a7d3e77eafe3dd8fad1d458d8e001fab/examples/community/imagic_stable_diffusion.py
"""

import logging
from typing import Any, List, Optional, Union

import numpy as np
import PIL.Image
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.diffusionmodules import UNetModel
from modules.encoders.autoencoder import AutoEncoder
from modules.encoders.textembedder import ClipTextEmbedder
from modules.sampler.ddim import DDIMSampler
from modules.sampler.plms import PLMSSampler
from src.modules.utils import get_device

logger = logging.getLogger(__name__)


class EmbeddingOptimization(pl.LightningModule):
    """
    Pipeline for optimizing the embedding of a given image.
    """

    def __init__(
        self,
        vae: AutoEncoder,
        text_embedder: ClipTextEmbedder,
        unet: UNetModel,
        scheduler: Union[DDIMSampler, PLMSSampler],
        embedding_learning_rate: float = 1e-3,
        text_embedding_optimization_steps: int = 500,
        height: int = 512,
        width: int = 512,
    ):
        super().__init__()
        self.vae: AutoEncoder = vae
        self.text_embedder: ClipTextEmbedder = text_embedder
        self.unet: UNetModel = unet
        self.scheduler: Union[DDIMSampler, PLMSSampler] = scheduler
        self.embedding_learning_rate = embedding_learning_rate
        self.text_embedding_optimization_steps = text_embedding_optimization_steps
        self.height = height
        self.width = width
        self.save_path = None

    def save_text_embeddings(self, path: str):
        """
        Save the optimized text optim to the given path.
        """
        text_embeddings = self.text_embeddings.detach().cpu().numpy()
        np.save(path, text_embeddings)

    def forward(self, x):
        pass

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.embedding_learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        prompt = batch["prompt"]

        text_input = self.text_embedder.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.text_embedder.tokenizer.model_max_length,
            return_tensors="pt",
            truncation=True,
        )
        text_embeddings = torch.nn.Parameter(
            self.text_embedder(text_input.input_ids.to(self.device))[0],
            requires_grad=True,
        ).detach()
        text_embeddings.requires_grad_(True)
        text_embeddings_orig = text_embeddings.clone()

        optimizer = self.optimizers()

        if self.height % 8 != 0 or self.width % 8 != 0:
            raise ValueError("Height and width must be divisible by 8.")

        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_embedder.requires_grad_(True)
        self.vae.eval()
        self.unet.eval()
        self.text_embedder.eval()

        image = image.to(self.device)
        init_latent_dist = self.vae.encode(image)
        image_latents = init_latent_dist.sample()
        image_latents = 0.18125 * image_latents

        progress_bar = tqdm(
            range(self.text_embedding_optimization_steps), disable=self.local_rank != 0
        )

        for _ in progress_bar:
            noise = torch.randn(image_latents.shape).to(image_latents.device)
            timesteps = torch.randint(1000, (1,), device=image_latents.device)

            noisy_latents = self.scheduler.add_noise(image_latents, noise, timesteps)
            noise_pred = self.unet(noisy_latents, timesteps, text_embeddings)

            loss = (
                F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            )
            self.log("loss", loss, on_step=True, on_epoch=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.text_embeddings_orig = text_embeddings_orig
        self.text_embeddings = text_embeddings

        if self.save_path:
            self.save_text_embeddings(self.save_path)
