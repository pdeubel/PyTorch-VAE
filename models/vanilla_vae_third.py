import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from torchvision import models

from models import BaseVAE
from .types_ import *


class VanillaVAE_third(BaseVAE):

    def __init__(self,
                 params: dict,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__(params)

        self.latent_dim = latent_dim
        self.teacher = models.vgg16(pretrained=True)

        _in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(_in_channels, 128, kernel_size=4, stride=2, padding=1),  # 128 => 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),  # 64 => 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),  # 32 => 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 16 => 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(512, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 200, kernel_size=8, stride=1),  # 8 => 1
            nn.Flatten(),
            Split())

        # TODO I hacked this hardcoded value into here because original author did the same, should be better done
        # # Build Decoder
        #
        # # TODO Same as above, redo the hardcoded stuff

        self.decoder = nn.Sequential(
            DeFlatten(),
            nn.ConvTranspose2d(100, 32, kernel_size=8, stride=1),  # 1 => 8
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(32, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 8 => 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # 16 => 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 32 => 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(128, _in_channels, kernel_size=4, stride=2, padding=1),  # 64 => 128
            nn.Identity(),
            nn.Sigmoid()
            # nn.Tanh()
        )

        self.adapter = nn.ModuleList([Adapter_model(128), Adapter_model(256), Adapter_model(512)])
        self.save_hyperparameters()

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        mu, log_var = self.encoder(input)

        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        # # TODO These hardcoded values should also be changed
        result = self.decoder(z)
        return result

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = log_var.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var, z

    def feature_extractor(self, x, model, target_layers):
        target_activations = list()
        for name, module in model._modules.items():
            x = module(x)
            if name in target_layers:
                target_activations += [x]
        return target_activations, x


    def training_step(self,  batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device
        output, input, mu, log_var, z = self.forward(real_img, labels=labels)

        s_activations, _ = self.feature_extractor(z, self.decoder, target_layers=['10', '16', '22'])
        t_activations, _ = self.feature_extractor(input, self.teacher.features, target_layers=['7', '14', '21'])
        train_loss = self.loss_function(output, input, mu, log_var, z,
                                        s_activations,
                                        t_activations,
                                        M_N=self.params['batch_size'] / self.num_train_imgs,
                                        optimizer_idx=optimizer_idx,
                                        batch_idx=batch_idx)

        # TODO this is deprecated
        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        output, input, mu, log_var, z = self.forward(real_img, labels=labels)
        s_activations, _ = self.feature_extractor(z, self.decoder, target_layers=['12', '18', '24'])
        t_activations, _ = self.feature_extractor(input, self.teacher.features, target_layers=['8', '15', '22'])
        val_loss = self.loss_function(output, input, mu, log_var, z, s_activations, t_activations,
                                      M_N=self.params['batch_size'] / self.num_val_imgs,
                                      optimizer_idx=optimizer_idx,
                                      batch_idx=batch_idx)

        return val_loss

    def loss_function(self, output, input, mu, log_var, z, s_activations, t_activations, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = output
        input = input
        mu = mu
        log_var = log_var
        # MSE_loss = nn.MSELoss(reduction='sum')

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)
        # recons_loss = MSE_loss(recons, input)

        mse_loss = 0
        for i in range(len(s_activations)):
            s_act = self.adapter[i](s_activations[-(i + 1)])
            mse_loss += F.mse_loss(s_act, t_activations[i])
            # mse_loss += MSE_loss(s_act, t_activations[i])

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
        # kld_loss = 0.5 * torch.sum(-1 - log_var + torch.exp(log_var) + mu ** 2)

        loss = recons_loss + kld_weight * kld_loss + mse_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss, 'mu': torch.mean(mu),
                'log_var': torch.mean(log_var), 'var': torch.mean(log_var.exp())}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
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

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class Adapter_model(nn.Module):
    def __init__(self, channel=128):
        super(Adapter_model, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, stride=1), nn.ReLU(),
                                  nn.Conv2d(channel, channel, kernel_size=1, stride=1))

    def forward(self, x):
        return self.conv(x)



class DeFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], 100, 1, 1)


class Split(nn.Module):
    def forward(self, x):
        mu, log_var = x.chunk(2, dim=1)
        return mu, log_var
