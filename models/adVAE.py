from typing import List, Any

import pytorch_lightning as pl
import torch
from torch import nn

from models import BaseVAE


class Encoder(pl.LightningModule):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None):
        super().__init__()

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        _in_channels = in_channels

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(_in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            _in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # TODO I hacked this hardcoded value into here because original author did the same, should be better done
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


class Decoder(pl.LightningModule):

    def __init__(self, latent_dim: int, hidden_dims: List = None):
        super().__init__()

        # Build Decoder
        modules = []

        # TODO Same as above, redo the hardcoded stuff
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

        hidden_dims.reverse()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # TODO These hardcoded values should also be changed
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


class Transformer(pl.LightningModule):

    def __init__(self, latent_dim: int):
        super().__init__()

        self.latent_dim = latent_dim

        self.transformer = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=2*self.latent_dim),
            nn.ELU())

    def forward(self, z: torch.Tensor) -> Any:
        z_T = self.transformer(z)
        z_T_mu = z_T[:, :self.latent_dim]
        z_T_sigma = torch.clip(z_T[:, self.latent_dim:], 1e-12, 1 - 1e-12)

        return z_T_mu, z_T_sigma


# noinspection PyPep8Naming
class adVAE(BaseVAE):

    def __init__(self,
                 params: dict,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__(params)

        self.latent_dim = latent_dim

        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim, hidden_dims=hidden_dims)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dims=hidden_dims)
        self.transformer = Transformer(latent_dim=latent_dim)

        self.save_hyperparameters()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training Step 1
        self.encoder.freeze()

        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        # TODO sigma or log_var
        z_T_mu, z_T_sigma = self.transformer(z)
        z_T = self.reparameterize(z_T_mu, z_T_sigma)

        x_r = self.decoder(z)
        x_T_r = self.decoder(z_T)

        mu_r, log_var_r = self.encoder(x_r)
        mu_T_r, sigma_T_r = self.encoder(x_T_r)



