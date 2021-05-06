from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from models import BaseVAE


class VanillaVAEUNet(BaseVAE):

    def __init__(self,
                 params: dict,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__(params)

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # TODO When debugging finished this and the hook registering can be deleted
        # def print_input_forward_hook(module, input, output):
        #     print(module)
        #
        #     print("-----------")
        #
        #     print("Input Shape: {}".format(input[0].shape))
        #     print("Output Shape: {}".format(output.shape))
        #
        #     print("-----------")
        #     return output

        # Build Encoder
        self.encoder_modules = nn.ModuleList()

        _in_channels = in_channels

        for h_dim in hidden_dims:
            sequential = nn.Sequential(
                nn.Conv2d(_in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            )

            # sequential.register_forward_hook(print_input_forward_hook)

            self.encoder_modules.append(sequential)
            _in_channels = h_dim

        # TODO I hacked this hardcoded value into here because original author did the same, should be better done
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        self.decoder_modules = nn.ModuleList()
        self.decoder_downsample_modules = nn.ModuleList()

        # TODO Same as above, redo the hardcoded stuff
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        # TODO change this to use [::-1] then remove this reverse. This is buggy because the model checkpointing
        #  saved the reverse hidden dims and this caused issues
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            sequential = nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i],
                                   hidden_dims[i + 1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU(),
            )

            # sequential.register_forward_hook(print_input_forward_hook)

            self.decoder_modules.append(sequential)

            self.decoder_downsample_modules.extend([
                nn.Conv2d(hidden_dims[i] * 2,
                          hidden_dims[i],
                          kernel_size=3,
                          padding=1),
                nn.LeakyReLU()
            ])

        self.decoder_modules.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU())
        )

        self.decoder_downsample_modules.extend([
            nn.Conv2d(hidden_dims[-1] * 2,
                      hidden_dims[-1],
                      kernel_size=3,
                      padding=1),
            nn.LeakyReLU()
        ])

        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[-1],
                      out_channels=3,
                      kernel_size=3,
                      padding=1),
            nn.Tanh()
        )

        # self.final_layer.register_forward_hook(print_input_forward_hook)
        hidden_dims.reverse()

        self.save_hyperparameters()

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        Saves the intermediate results in a list so that the decoder of the UNet can use them.

        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        x = input

        # TODO this could be a problem when the memory is not freed immediately on the GPU. A solution would be to
        #  explicitly delete the list and Tensors
        encoder_results = []

        for encoder_layer in self.encoder_modules:
            x = encoder_layer(x)

            encoder_results.append(x)

        result = x

        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var, encoder_results]

    def decode(self, z: torch.Tensor, encoder_results: list = None) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.

        Concatenates the corresponding result from the encoder layer to each decoder layer. Then Conv2D layers are used
        to sample down the number of channels again.

        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # TODO These hardcoded values should also be changed
        result = result.view(-1, 512, 2, 2)

        x = result

        if encoder_results is not None:
            decoder_iteration = zip(self.decoder_modules, self.decoder_downsample_modules, encoder_results[::-1])

            for decoder_layer, downsample_module, encoder_layer_result in decoder_iteration:
                x = torch.cat([encoder_layer_result, x], dim=1)
                x = downsample_module(x)
                x = decoder_layer(x)
        else:
            for decoder_layer in self.decoder_modules:
                x = decoder_layer(x)

        result = self.final_layer(x)

        return result

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

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var, encoder_results = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z, encoder_results), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss, 'mu': torch.mean(mu),
                'log_var': torch.mean(log_var), 'var': torch.mean(log_var.exp())}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
