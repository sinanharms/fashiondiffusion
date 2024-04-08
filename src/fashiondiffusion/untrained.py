import os
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import utils
from torchvision.transforms import transforms

from src.modules.diffusionmodules.unet import UNetModel
from src.modules.utils import get_device
from src.scheduler.ddim import DDIMSampler


def generate(
    prompt: str,
    in_channels: int,
    img_size: int,
    n_timesteps: int,
    n_inference_timesteps: int,
    batch_size: int,
    pre_trained_model_path: str,
):
    unet = UNetModel(
        in_channels=in_channels, out_channels=in_channels, image_size=img_size
    )
    pretrained = torch.load(pre_trained_model_path)
    unet.load_state_dict(pretrained, strict=False)

    noise_scheduler = DDIMSampler(
        num_train_timesteps=n_timesteps, beta_schedule="cosine"
    )

    device = get_device()
    unet = unet.to(device)

    with torch.no_grad():
        generator = torch.manual_seed(0)
        samples = noise_scheduler.sample(
            model=unet,
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=n_inference_timesteps,
            eta=0.5,
            use_clipped_model_output=True,
            output_type="numpy",
        )
        images = samples["sample"]
        images_processed = (images * 255).round().astype("uint8")

        generated_images = []
        for i, image in enumerate(images_processed):
            image = Image.fromarray(image)
            generated_images.append(image)

        grid = utils.make_grid(samples["sample_tensor"], nrow=batch_size // 4)
        grid = transforms.ToPILImage()(grid)

        return generated_images, grid
