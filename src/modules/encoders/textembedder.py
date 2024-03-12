import clip
import torch
import torch.nn as nn
from einops import repeat


class TextEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClipTextEmbedder(TextEmbedder):
    """
    A wrapper around OpenAI's CLIP model for text embedding.
    """

    def __init__(
        self,
        version,
        device="cuda",
        max_length=77,
        n_repeat=1,
        normalize=True,
        pretrained_weights=None,
    ):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

        if pretrained_weights:
            self.model.load_state_dict(
                torch.load(pretrained_weights, map_location="cpu")
            )

    def freeze(self):
        self.model = self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = repeat(z, "b 1 d -> b k d", k=self.n_repeat)
        return z
