from typing import Any, List

import torch
from torch import nn
from torch.nn import functional as F

from modules.encoders.base import BaseVAE, Torch


class SimpleVAE(BaseVAE):
    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.encoder = nn.ModuleList([])
        encoder_layers = []
        for hdim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=hdim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hdim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = hdim

        self.encoder.append(nn.Sequential(*encoder_layers))
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.decoder(nn.ModuleList([]))
        self.decoder_input(nn.Linear(latent_dim, hidden_dims[-1] * 4))

        hidden_dims.reverse()
        decoder_layers = []
        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder.append(nn.Sequential(*decoder_layers))

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: Torch, **kwargs) -> List[Torch]:
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Torch, **kwargs) -> Any:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Torch, log_var: Torch) -> Torch:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Torch) -> List[Torch]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(mu, sigma), N(0, 1)) = log frac{1}{sigma} + frac{sigma^2 + mu^2}{2} - frac{1}{2}
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recon_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )
        loss = recon_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "Reconstruction_loss": recon_loss.detach(),
            "KLDiv": -recon_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Torch:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x: Torch, **kwargs) -> Torch:
        return self.forward(x)[0]
