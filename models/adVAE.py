from typing import List, Any

import pytorch_lightning as pl
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F

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

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param log_var: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        return [mu, log_var, z]

    def forward(self, x: torch.Tensor, optimizer_idx=0) -> dict:
        # TODO maybe refine this so some computations don't have to be executed twice per batch
        #
        # equal = torch.equal(self.encoder.state_dict()["encoder.0.0.weight"], self.last_x)
        # self.last_x = self.encoder.state_dict()["encoder.0.0.weight"].clone()

        # Training Step 1
        mu, log_var, z = self.encode(x)

        mu_T, log_var_T = self.transformer(z)
        z_T = self.reparameterize(mu_T, log_var_T)

        x_r = self.decoder(z)
        x_T_r = self.decoder(z_T)

        mu_r, log_var_r, _ = self.encode(x_r)
        mu_T_r, log_var_T_r, _ = self.encode(x_T_r)

        return {"x": x,
                "x_r": x_r,
                "x_T_r": x_T_r,
                "mu": mu,
                "log_var": log_var,
                "mu_T": mu_T,
                "log_var_T": log_var_T,
                "mu_r": mu_r,
                "log_var_r": log_var_r,
                "mu_T_r": mu_T_r,
                "log_var_T_r": log_var_T_r}

        # # Training Step 2
        # # self.decoder.freeze()
        # # self.transformer.freeze()
        #
        # # TODO Could be that x_r and x_T_r need to be detached
        # x_r = self.decoder(self.z)
        # x_T_r = self.decoder(self.z_T)
        #
        # mu_r, log_var_r = self.encoder(x_r)
        # mu_T_r, log_var_T_r = self.encoder(x_T_r)
        #
        # # self.decoder.unfreeze()
        # # self.transformer.unfreeze()
        #
        # # Loss Step 2
        # return {"x": x,
        #         "x_r": x_r,
        #         "mu": self.mu,
        #         "log_var": self.log_var,
        #         "mu_r": mu_r,
        #         "log_var_r": log_var_r,
        #         "mu_T_r": mu_T_r,
        #         "log_var_T_r": log_var_T_r}

    def loss_function(self,
                      results,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        optimizer_idx = kwargs['optimizer_idx']

        if optimizer_idx == 0:
            # Update Generator and Transformer
            x = results["x"]
            x_r = results["x_r"]
            x_T_r = results["x_T_r"]
            mu = results["mu"]
            log_var = results["log_var"]
            mu_T = results["mu_T"]
            log_var_T = results["log_var_T"]
            mu_r = results["mu_r"]
            log_var_r = results["log_var_r"]
            mu_T_r = results["mu_T_r"]
            log_var_T_r = results["log_var_T_r"]

            L_G_z_term_1 = F.mse_loss(x, x_r)
            L_G_z_term_2 = torch.mean(-0.5 * torch.sum(1 + log_var_r - mu_r ** 2 - log_var_r.exp(), dim=1), dim=0)
            L_G_z = L_G_z_term_1 + self.params["gamma"] * L_G_z_term_2

            L_G_z_T_term_1 = F.relu(self.params["m_x"] - F.mse_loss(x_r, x_T_r))
            L_G_z_T_term_2 = F.relu(self.params["m_z"] - torch.mean(-0.5 * torch.sum(1 + log_var_T_r - mu_T_r ** 2 - log_var_T_r.exp(), dim=1), dim=0))
            L_G_z_T = L_G_z_T_term_1 + self.params["gamma"] * L_G_z_T_term_2

            L_G = L_G_z + L_G_z_T

            L_T_term_1 = 0.5 * log_var_T - 0.5 * log_var
            L_T_term_2 = (log_var.exp() + torch.square(mu - mu_T)) / (2 * log_var_T.exp())
            L_T_term_3 = -0.5

            L_T = torch.mean(torch.sum(L_T_term_1 + L_T_term_2 + L_T_term_3, dim=1), dim=0)

            loss = L_T + self.params["gamma"] * L_G

            return {"loss": loss, "loss_G": L_G, "loss_G_z": L_G_z, "loss_G_z_T": L_G_z_T, "loss_T": L_T,
                    "l_G_z_term_1": L_G_z_term_1, "l_G_z_term_2": L_G_z_term_2,
                    "l_G_z_T_term_1": L_G_z_T_term_1, "l_G_z_T_term_2": L_G_z_T_term_2,
                    "l_T_term_1": torch.mean(torch.sum(L_T_term_1, dim=1), dim=0),
                    "l_T_term_2": torch.mean(torch.sum(L_T_term_2, dim=1), dim=0),
                    "mu": torch.mean(mu), "var": torch.mean(log_var.exp()),
                    "mu_T": torch.mean(mu_T), "var_T": torch.mean(log_var_T.exp())}
        elif optimizer_idx == 1:
            # Update Encoder
            x = results["x"]
            x_r = results["x_r"]
            mu = results["mu"]
            log_var = results["log_var"]
            mu_r = results["mu_r"]
            log_var_r = results["log_var_r"]
            mu_T_r = results["mu_T_r"]
            log_var_T_r = results["log_var_T_r"]

            L_E_term_1 = F.mse_loss(x, x_r)
            L_E_term_2 = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            L_E_term_3 = F.relu(self.params["m_z"] - torch.mean(-0.5 * torch.sum(1 + log_var_r - mu_r ** 2 - log_var_r.exp(), dim=1), dim=0))
            L_E_term_4 = F.relu(self.params["m_z"] - torch.mean(-0.5 * torch.sum(1 + log_var_T_r - mu_T_r ** 2 - log_var_T_r.exp(), dim=1), dim=0))
            L_E = L_E_term_1 + self.params["lambda"] * L_E_term_2 + self.params["gamma"] * L_E_term_3 + self.params["gamma"] * L_E_term_4

            loss = L_E

            return {"loss": loss, "loss_E": L_E,
                    "l_E_term_1": L_E_term_1, "l_E_term_2": L_E_term_2, "l_E_term_3": L_E_term_3,
                    "l_E_term_4": L_E_term_4}

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, optimizer_idx=optimizer_idx)
        train_loss = self.loss_function(results,
                                        M_N=self.params['batch_size'] / self.num_train_imgs,
                                        optimizer_idx=optimizer_idx,
                                        batch_idx=batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, optimizer_idx=optimizer_idx)
        val_loss = self.loss_function(results,
                                      M_N=self.params['batch_size'] / self.num_val_imgs,
                                      optimizer_idx=optimizer_idx,
                                      batch_idx=batch_idx)

        return val_loss

    def configure_optimizers(self):
        """
        Configures two optimizers, the first for the decoder and transformer and the second for the encoder.
        This means that training_step and validation_step are called twice for each mini-batch, once per optimizer.

        Each optimizer only changes the weights which it received in the constructor below.
        """
        decoder_transformer_optimizer = optim.Adam(
            [{"params": self.decoder.parameters()}, {"params": self.transformer.parameters()}],
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
        )

        encoder_optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
        )

        return [decoder_transformer_optimizer, encoder_optimizer]

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.Tensor:
        z = torch.randn(batch_size, self.latent_dim).to(current_device)

        return self.decoder(z)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        mu, log_var, z = self.encode(x)
        # reconstructions = []
        # for i in range(10):
        #     current_z = self.reparameterize(mu, log_var)
        #
        #     reconstructions.append(self.decoder(current_z))

        return self.decoder(z)

