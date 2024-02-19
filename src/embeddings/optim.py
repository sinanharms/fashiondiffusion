import inspect
import logging
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
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

    def train(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image],
        height: Optional = 512,
        width: Optional = 512,
        generator: Optional[torch.Generator] = None,
        embedding_learning_rate: Optional[float] = 1e-3,
        text_embedding_optimization_steps: Optional[int] = 500,
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
            pass

        self.text_embeddings_orig = text_embeddings_orig
        self.text_embeddings = text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        alpha: float = 1.2,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
    ):
        """
        Generate an image from the optimized text embedding.
        """
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )
        if self.text_embeddings is None:
            raise ValueError(
                "Please run the pipe.train() before trying to generate an image."
            )
        if self.text_embeddings_orig is None:
            raise ValueError(
                "Please run the pipe.train() before trying to generate an image."
            )
        text_embeddings = (
            alpha * self.text_embeddings_orig + (1 - alpha) * self.text_embeddings
        )
        # expand the text embeddings if we are doing classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latens_shape = (1, self.unet.config.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if self.device == "mps":
            latents = torch.randn(
                latens_shape, generator=generator, dtype=latents_dtype, device="cpu"
            ).to(self.device)
        else:
            latents = torch.randn(
                latens_shape,
                generator=generator,
                dtype=latents_dtype,
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        latents = latents * self.scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pt"
            ).to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype),
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return image, has_nsfw_concept

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )
