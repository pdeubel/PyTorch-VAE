from typing import List, Any

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
from torch.nn import functional as F

from models import BaseVAE


def kl_divergence(mu, log_var):
    return 0.5 * torch.sum(-log_var - 1 + log_var.exp() + torch.square(mu), dim=1)


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
            nn.Linear(in_features=self.latent_dim, out_features=4 * self.latent_dim),
            nn.LeakyReLU()
        )

        self.mu_T = nn.Linear(in_features=4 * self.latent_dim, out_features=self.latent_dim)
        self.log_var_T = nn.Linear(in_features=4 * self.latent_dim, out_features=self.latent_dim)

    def forward(self, z: torch.Tensor) -> Any:
        z_T = self.transformer(z)
        z_T_mu = self.mu_T(z_T)
        z_T_log_var = self.log_var_T(z_T)

        return z_T_mu, z_T_log_var


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

        # Only used for logging to see if training converges
        self.summary_loss = 0

        self.save_hyperparameters()

        self.mu, self.log_var = None

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
        self.mu, self.log_var, z = self.encode(x)

        mu_T, log_var_T = self.transformer(z)
        z_T = self.reparameterize(mu_T, log_var_T)

        x_r = self.decoder(z)
        x_T_r = self.decoder(z_T)

        mu_r, log_var_r, _ = self.encode(x_r)
        mu_T_r, log_var_T_r, _ = self.encode(x_T_r)

        return {"x": x,
                "x_r": x_r,
                "x_T_r": x_T_r,
                "mu": self.mu,
                "log_var": self.log_var,
                "mu_T": mu_T,
                "log_var_T": log_var_T,
                "mu_r": mu_r,
                "log_var_r": log_var_r,
                "mu_T_r": mu_T_r,
                "log_var_T_r": log_var_T_r}

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

        if optimizer_idx == 0:
            L_T_term_1 = 0.5 * log_var_T - 0.5 * log_var
            L_T_term_2 = (log_var.exp() + torch.square(mu - mu_T)) / (2 * log_var_T.exp())
            L_T_term_3 = -0.5

            L_T = torch.sum(L_T_term_1 + L_T_term_2 + L_T_term_3, dim=1)

            L_G_z_term_1 = F.mse_loss(x, x_r)
            L_G_z_term_2 = kl_divergence(mu_r, log_var_r)
            L_G_z = L_G_z_term_1 + kld_weight * L_G_z_term_2

            L_G_z_T_term_1 = F.relu(self.params["m_x"] - F.mse_loss(x_r, x_T_r))
            L_G_z_T_term_2 = F.relu(self.params["m_z"] - kl_divergence(mu_T_r, log_var_T_r))

            L_G_z_T = L_G_z_T_term_1 + L_G_z_T_term_2

            L_G = L_G_z + self.params["gamma"] * L_G_z_T

            loss = torch.mean((self.params["gamma"] * L_T) + (self.params["lambda_G"] * L_G), dim=0)

            # This is used for logging
            loss_T = torch.mean(self.params["gamma"] * L_T, dim=0)
            loss_G = torch.mean(self.params["lambda_G"] * L_G, dim=0)

            # This is used when loss_function is called with optimizer_idx=1, then the encoder loss is added and
            # the summed loss of generator+transformer and encoder is logged
            self.summary_loss = loss

            return {"loss": loss,
                    "loss_T": loss_T, "loss_G": loss_G}

        elif optimizer_idx == 1:
            # Update Encoder
            L_E_term_1 = F.mse_loss(x, x_r)
            L_E_term_2 = kl_divergence(mu, log_var)
            L_E_term_3 = F.relu(self.params["m_z"] - kl_divergence(mu_r, log_var_r))
            L_E_term_4 = F.relu(self.params["m_z"] - kl_divergence(mu_T_r, log_var_T_r))

            L_E = L_E_term_1 + kld_weight * L_E_term_2 + self.params["gamma"] * L_E_term_3 + self.params[
                "gamma"] * L_E_term_4

            loss = torch.mean(L_E, dim=0)

            return {"loss": loss,
                    "loss_summary": self.summary_loss + loss}

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

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        if self.current_epoch % 1 == 0 or self.current_epoch == (self.trainer.max_epochs - 1):
            self.sample_images()

        self.logger.experiment.log({'avg_val_loss': avg_loss})

        return {'val_loss': avg_loss}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        recons, synthesized_anomalies = self.generate(test_input, labels=test_label)
        samples_normal, samples = self.sample(self.params["batch_size"], self.curr_device, labels=test_label)

        self.logger.experiment.add_images("originals",
                                          test_input,
                                          global_step=self.current_epoch)

        self.logger.experiment.add_images("reconstructions",
                                          recons,
                                          global_step=self.current_epoch)

        self.logger.experiment.add_images("synthesized anomalies",
                                          synthesized_anomalies,
                                          global_step=self.current_epoch)

        self.logger.experiment.add_images("samples_normal",
                                          samples_normal,
                                          global_step=self.current_epoch)

        self.logger.experiment.add_images("samples",
                                          samples,
                                          global_step=self.current_epoch)

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

    def sample(self, batch_size: int, current_device: int, **kwargs):
        z_normal = torch.randn(batch_size, self.latent_dim).to(current_device)

        distribution = Normal(loc=torch.mean(torch.sum(self.mu, dim=1), dim=0),
                              scale=torch.mean(torch.sum(self.sigma.exp(), dim=1), dim=0))

        z = distribution.sample((batch_size, self.latent_dim)).to(current_device)

        return self.decoder(z_normal), self.decoder(z)

    def generate(self, x: torch.Tensor, **kwargs):
        results = self.forward(x)

        return results["x_r"], results["x_T_r"]
