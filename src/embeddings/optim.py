"""
Taken and modified from:
 https://github.com/huggingface/diffusers/blob/12004bf3a7d3e77eafe3dd8fad1d458d8e001fab/examples/community/imagic_stable_diffusion.py
"""

import logging
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def preprocess_image(image: PIL.Image.Image):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class EmbeddingOptimizationPipeline(DiffusionPipeline):
    """
    Pipeline for optimizing the embedding of a given image.
    """

    def __init__(
        self, vae, text_encoder, tokenizer, unet, scheduler, feature_extractor
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        """
        Enable attention slicing for the text encoder.
        """
        if slice_size == "auto":
            slize_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slicing(slice_size)

    def disable_attention_slicing(self):
        """
        Disable attention slicing for the text encoder.
        """
        self.unet.disable_attention_slicing()

    def save_text_embeddings(self, path: str):
        """
        Save the optimized text embeddings to the given path.
        """
        text_embeddings = self.text_embeddings.detach().cpu().numpy()
        np.save(path, text_embeddings)

    def train(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image],
        height: Optional = 512,
        width: Optional = 512,
        generator: Optional[torch.Generator] = None,
        embedding_learning_rate: Optional[float] = 1e-3,
        text_embedding_optimization_steps: Optional[int] = 500,
        save_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Optimize the embedding of the given image to match the given prompt.

        """
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16",
        )
        if torch.device in kwargs:
            device = kwargs.pop("torch_device")

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("Height and width must be divisible by 8.")

        # freeze the VAE and the unet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(True)
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

        if accelerator.is_main_process:
            accelerator.init_trackers(
                "optimization",
                config={
                    "embedding_learning_rate": embedding_learning_rate,
                    "text_embedding_optimization_steps": text_embedding_optimization_steps,
                },
            )

        # get text embeddings for prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            truncation=True,
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0],
            requires_grad=True,
        )
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_(True)
        text_embeddings_orig = text_embeddings.clone()

        # init optimizer
        optimizer = torch.optim.Adam([text_embeddings], lr=embedding_learning_rate)

        if isinstance(image, PIL.Image.Image):
            image = preprocess_image(image)

        latents_dtype = text_embeddings.dtype
        image = image.to(self.device, dtype=latents_dtype)
        init_latent_dist = self.vae.encode(image).latent_dist
        image_latents = init_latent_dist.sample(generator=generator)
        image_latents = 0.18125 * image_latents

        progress_bar = tqdm(
            range(text_embedding_optimization_steps),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        global_step = 0

        logger.info("Optimizing text embedding.")
        for _ in range(text_embedding_optimization_steps):
            with accelerator.accumulate(text_embeddings):
                # sample noise that will add to latents
                noise = torch.randn(image_latents.shape).to(image_latents.device)
                timesteps = torch.randint(1000, (1,), device=image_latents.device)

                # add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = self.scheduler.add_noise(
                    image_latents, noise, timesteps
                )

                # predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item()
            }  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            accelerator.wait_for_everyone()

            text_embeddings.requires_grad_(False)

        self.text_embeddings_orig = text_embeddings_orig
        self.text_embeddings = text_embeddings

        if save_path:
            self.save_text_embeddings(save_path)

    def __call__(
        self, dataset: torch.utils.data.Dataset, batch_size: int, *args, **kwargs
    ):
        """
        Optimize the text embeddings of the images in the given dataset.
        """

        dataloader = DataLoader(dataset, batch_size=batch_size, **kwargs)

        for batch in dataloader:
            images = batch["image"]
            if "prompt" in batch:
                prompt = batch["prompt"]

                self.train(prompt=prompt, image=images)
