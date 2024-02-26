import clip
import kornia
import torch
import torch.nn as nn


class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class CLIPImageEmbedder(ImageEmbedder):
    """
    A wrapper around OpenAI's CLIP model for image embedding.
    """

    def __init__(
        self,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        jit=False,
        antialias=False,
    ):
        super().__init__()
        self.model, _ = clip.load(model, device=device, jit=jit)
        self.anti_alias = antialias

        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )

    def preprocess(self, x):
        # normalize to [0, 1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.anti_alias,
        )
        x = (x + 1.0) / 2.0

        # renormalize according to CLIP's requirements
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # preprocess the image, x assumed in range [-1, 1]
        return self.model.encode_image(self.preprocess(x))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
