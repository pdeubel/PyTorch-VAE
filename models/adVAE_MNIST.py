"""
MIT License

Copyright (c) 2019 YeongHyeon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
PyTorch implementation of adVAE for MNIST. This is adapted from the following TensorFlow implementation:
https://github.com/YeongHyeon/adVAE 
"""

from typing import List, Any

import pytorch_lightning as pl
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F

from models import BaseVAE


class Encoder(pl.LightningModule):

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3), stride=1,
                               padding=1)

        self.activation = nn.ELU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1,
                               padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1,
                               padding=1)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1,
                               padding=1)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1,
                               padding=1)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,
                               padding=1)

        self.fc1 = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=latent_dim * 2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.max_pool(x)

        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.max_pool(x)

        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))

        x = torch.flatten(x, start_dim=1)

        x = self.activation(self.fc1(x))
        z_params = self.fc2(x)

        z_mu = z_params[:, :self.latent_dim]
        z_log_var = z_params[:, self.latent_dim:]

        return [z_mu, z_log_var]


class Decoder(pl.LightningModule):

    def __init__(self, latent_dim: int):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(in_features=self.latent_dim, out_features=512)
        self.activation = nn.ELU()
        # out_features is the output shape of the last conv layer in the encoder
        self.fc2 = nn.Linear(in_features=512, out_features=(64 * 7 * 7))

        # They seem to use
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1,
                               padding=1)

        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=2,
                                                  padding=1, output_padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1,
                               padding=1)

        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=2,
                                                  padding=1, output_padding=1)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1,
                               padding=1)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=1,
                               padding=1)

        self.output_activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        x = self.activation(self.fc1(z))
        x = self.activation(self.fc2(x))
        x = x.view(-1, 64, 7, 7)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = self.activation(self.conv_transpose1(x))
        x = self.activation(self.conv3(x))

        x = self.activation(self.conv_transpose2(x))
        x = self.activation(self.conv4(x))
        x = self.output_activation(self.conv5(x))

        return x


class Transformer(pl.LightningModule):

    def __init__(self, latent_dim: int):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(in_features=self.latent_dim, out_features=2 * self.latent_dim)
        self.activation = nn.ELU()

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        z_params = self.activation(self.fc1(z))

        z_T_mu = z_params[:, :self.latent_dim]
        z_T_log_var = z_params[:, self.latent_dim:]

        return [z_T_mu, z_T_log_var]


# noinspection PyPep8Naming
class adVAEMNIST(BaseVAE):

    def __init__(self,
                 params: dict,
                 in_channels: int,
                 latent_dim: int,
                 **kwargs) -> None:
        super().__init__(params)

        self.latent_dim = latent_dim

        self.lambda_T = 1e-22
        self.lambda_G = 10.0

        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.transformer = Transformer(latent_dim=latent_dim)

        # Only used for logging to see if training converges
        self.summary_loss = 0

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
            L_G_z = L_G_z_term_1 + L_G_z_term_2

            L_G_z_T_term_1 = self.params["m_x"] - F.mse_loss(x_r, x_T_r)
            L_G_z_T_term_1 = torch.maximum(torch.zeros_like(L_G_z_T_term_1), L_G_z_T_term_1)

            L_G_z_T_term_2 = self.params["m_z"] - torch.mean(
                -0.5 * torch.sum(1 + log_var_T_r - mu_T_r ** 2 - log_var_T_r.exp(), dim=1), dim=0)
            L_G_z_T_term_2 = torch.maximum(torch.zeros_like(L_G_z_T_term_2), L_G_z_T_term_2)

            L_G_z_T = L_G_z_T_term_1 + L_G_z_T_term_2

            L_G = L_G_z + L_G_z_T

            L_T_term_1 = 0.5 * log_var_T - 0.5 * log_var
            L_T_term_2 = (log_var.exp() + torch.square(mu - mu_T)) / (2 * log_var_T.exp())
            L_T_term_3 = -0.5

            L_T = torch.sum(L_T_term_1 + L_T_term_2 + L_T_term_3, dim=1)

            loss = torch.mean((self.lambda_T * L_T) + (self.lambda_G * L_G), dim=0)

            # This is used when loss_function is called with optimizer_idx=1, then the encoder loss is added and
            # the summed loss of generator+transformer and encoder is logged
            self.summary_loss = loss

            return {"loss": loss, "loss_G_and_T": loss, "loss_G": L_G, "loss_G_z": L_G_z, "loss_G_z_T": L_G_z_T,
                    "loss_T": torch.mean(L_T, dim=0),
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
            L_E_term_3 = self.params["m_z"] - torch.mean(-0.5 * torch.sum(1 + log_var_r - mu_r ** 2 - log_var_r.exp(), dim=1), dim=0)
            L_E_term_3 = torch.maximum(torch.zeros_like(L_E_term_3), L_E_term_3)
            L_E_term_4 = self.params["m_z"] - torch.mean(-0.5 * torch.sum(1 + log_var_T_r - mu_T_r ** 2 - log_var_T_r.exp(), dim=1), dim=0)
            L_E_term_4 = torch.maximum(torch.zeros_like(L_E_term_4), L_E_term_4)
            L_E = L_E_term_1 + L_E_term_2 + L_E_term_3 + L_E_term_4

            loss = torch.mean(L_E, dim=0)

            return {"loss": loss, "loss_summary": self.summary_loss + loss, "loss_E": L_E,
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

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        if self.current_epoch % 1 == 0 or self.current_epoch == (self.trainer.max_epochs - 1):
            self.sample_images()

        self.logger.experiment.log({'avg_val_loss': avg_loss})

        return {'val_loss': avg_loss}

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
