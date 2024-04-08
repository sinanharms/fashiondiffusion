from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from src.modules.encoders.base import BaseVAE, Torch


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Torch) -> Torch:
        latents = latents.permute(
            0, 2, 3, 1
        ).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)

        # compute L2 distance between latents and embedding
        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )

        # get encoding that has minimum distance
        encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)

        # convert to one-hot
        encoding_one_hot = torch.zeros(
            encoding_indices.shape[0], self.K, device=latents.device
        )
        encoding_one_hot.scatter_(1, encoding_indices, 1)

        # quantize latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view(latents_shape)

        # compute loss for embedding
        loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = embedding_loss + self.beta * loss

        # Add residue to latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Torch) -> Torch:
        return x + self.resblock(x)


class VQVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        num_embeddings: int,
        embedding_dim: int,
        hidden_dims: List[int],
        beta: float,
        img_size: int,
        **kwargs,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.img_size = img_size

        if hidden_dims is None:
            hidden_dims = [128, 256]
        # Build Encoder
        self.encoder = nn.ModuleList([])
        encoder_layers = []
        for h_dim in hidden_dims:
            encoder_layers = [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            ]
            in_channels = h_dim

        encoder_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        )

        for _ in range(6):
            encoder_layers.append(ResidualLayer(in_channels, in_channels))
        encoder_layers.append(nn.LeakyReLU())

        encoder_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels=embedding_dim, kernel_size=1, stride=1
                ),
                nn.LeakyReLU(),
            )
        )

        self.encoder.append(nn.Sequential(*encoder_layers))
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, beta)

        # Build Decoder
        self.decoder = nn.ModuleList([])
        decoder_layers = [
            nn.Sequential(
                nn.Conv2d(
                    embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1
                ),
                nn.LeakyReLU(),
            )
        ]

        for _ in range(6):
            decoder_layers.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        decoder_layers.append(nn.LeakyReLU())

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    out_channels=in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),
            )
        )

        self.decoder.append(nn.Sequential(*decoder_layers))

    def encode(self, x: Torch, **kwargs) -> List[Torch]:
        x = self.encoder(x)
        return [x]

    def decode(self, z: Torch, **kwargs) -> Any:
        return self.decoder(z)

    def forward(self, x: Torch) -> List[Torch]:
        encoding = self.encode(x)[0]
        quantized_encoding, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_encoding), x, vq_loss]

    def loss(self, *args, **kwargs) -> Dict:
        reconstruction = args[0]
        input = args[1]
        vq_loss = args[2]
        reconstruction_loss = F.mse_loss(reconstruction, input)

        loss = reconstruction_loss + vq_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": reconstruction_loss,
            "VQ_Loss": vq_loss,
        }

    def sample(self, z, batch_size: int, device: str, **kwargs) -> Torch:
        raise Warning("Sample method not implemented for VQVAE")

    def generate(self, x: Torch, **kwargs) -> Torch:
        return self.forward(x)[0]
