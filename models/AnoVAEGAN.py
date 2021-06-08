import torch
from torch import nn
from abc import abstractmethod
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import roc_curve, auc
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms
from .types_ import List, Tensor
from datasets.concrete_cracks import ConcreteCracksDataset
from datasets.sdnet2018 import SDNet2018

class AnoVAEGAN(pl.LightningModule):
    def __init__(self,
                 params: dict,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(AnoVAEGAN, self).__init__()
        self.params = params
        self.curr_device = None
        self.noise_type = kwargs['noise_type']
        self.noise = kwargs['noise']

        try:
            num_workers = params["dataloader_workers"]
        except KeyError:
            num_workers = 1

        self.additional_dataloader_args = {'num_workers': num_workers, 'pin_memory': True}

        self.latent_dim = latent_dim
        # self.anovae = AnoVAE(in_channels, latent_dim, hidden_dims, **kwargs)
        self.Encoder = Encoder(in_channels, latent_dim, hidden_dims, **kwargs)
        self.Decoder = Decoder(in_channels, latent_dim, hidden_dims, **kwargs)
        self.discriminator = Discriminator(in_channels, latent_dim, hidden_dims, **kwargs)
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        z, x, mu, log_var = self.Encoder(x)
        out = self.Decoder(z)
        return out

    def vae_step(self, x: torch.Tensor) -> dict:
        z, input, mu, log_var = self.Encoder(x)
        recons = self.Decoder(z)
        dis = self.discriminator(recons)
        rec_loss = nn.L1Loss(reduction='mean')
        adv_loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        valid = torch.full((x.shape[0], 1), 1, dtype=torch.float).to(self.curr_device)

        rec_weight = 1  # kwargs[]
        kld_weight = 1  # kwargs['M_N']  # Account for the minibatch samples from the dataset
        adv_weight = 1

        # recons_loss = F.mse_loss(recons, input)
        recons_loss = rec_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        adv_loss = adv_loss_function(dis, valid)

        loss = rec_weight * recons_loss + kld_weight * kld_loss + adv_weight * adv_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss, 'mu': torch.mean(mu), 'Adl_loss': adv_loss,
                'log_var': torch.mean(log_var), 'var': torch.mean(log_var.exp())}

    def generator_step(self, x: torch.Tensor) -> dict:
        z, x, mu, log_var = self.Encoder(x)
        out = self.Decoder(z.detach())
        dis = self.discriminator(out)
        valid = torch.full((x.shape[0], 1), 1, dtype=torch.float).to(self.curr_device)

        # rec_weight = 1  # kwargs[]
        # kld_weight = 1  # kwargs['M_N']  # Account for the minibatch samples from the dataset
        # adv_weight = 1

        adv_loss_function = nn.BCEWithLogitsLoss(reduction='mean')
        # rec_loss = nn.L1Loss(reduction='mean')

        # recons_loss = F.mse_loss(recons, input)
        # recons_loss = rec_loss(recons, input)
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # adv_loss = -torch.mean(dis)
        # adv_loss = -torch.mean(torch.log(dis + 1e-6 * torch.ones(dis.shape).to(self.curr_device)))
        adv_loss = adv_loss_function(dis, valid)

        # loss = rec_weight * recons_loss + kld_weight * kld_loss + adv_weight * adv_loss
        loss = adv_loss

        return {'loss': loss,  'Adl_loss': adv_loss}

    def discriminator_step(self, x: torch.Tensor) -> dict:
        adv_loss_function = nn.BCEWithLogitsLoss(reduction='mean')

        real_img = self.discriminator(x)
        fake_img = self.discriminator(self.forward(x).detach())

        valid = torch.full((x.shape[0], 1), 1, dtype=torch.float).to(self.curr_device)
        fake = torch.full((x.shape[0], 1), 0, dtype=torch.float).to(self.curr_device)

        real_loss = adv_loss_function(real_img, valid)
        fake_loss = adv_loss_function(fake_img, fake)
        # real_loss = -torch.mean(torch.log(real_img + 1e-6 * torch.ones(real_img.shape).to(self.curr_device)))
        # fake_loss = -torch.mean(torch.log(torch.ones(fake_img.shape).to(self.curr_device)-fake_img + 1e-6 * torch.ones(fake_img.shape).to(self.curr_device)))
        # disc_loss = torch.mean(fake_img) - torch.mean(real_img)
        disc_loss = real_loss + fake_loss
        # tqdm_dict = {'disc_loss': disc_loss}

        return {'loss': disc_loss, 'disc_loss': disc_loss}

    def training_step(self, batch, batch_idx, optimizer_idx):
        X, labels = batch
        self.curr_device = X.device
        if self.noise_type == 'gaussian':
            noise_img = self.add_noise(X)
        elif self.noise_type == 'mask':
            noise_img = self.mask(X)

        # train vae
        if optimizer_idx == 0:
            train_loss = self.vae_step(noise_img)
            self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        # train generator
        if optimizer_idx == 1:
            train_loss = self.generator_step(noise_img)
            self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        # train discriminator
        if optimizer_idx == 2:
            train_loss = self.discriminator_step(noise_img)
            self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        X, labels = batch
        self.curr_device = X.device
        if self.noise_type == 'gaussian':
            noise_img = self.add_noise(X)
        elif self.noise_type == 'mask':
            noise_img = self.mask(X)

        # train vae
        if optimizer_idx == 0:
            val_loss = self.vae_step(noise_img)
        # train generator
        if optimizer_idx == 1:
            val_loss = self.generator_step(noise_img)
        # train discriminator
        if optimizer_idx == 2:
            val_loss = self.discriminator_step(noise_img)

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

        if self.current_epoch % 5 == 0:
            self.sample_images()

        if self.current_epoch == (self.trainer.max_epochs - 1):
            # We are in the last epoch, calculate ROC and AUC and log to tensorboard
            self.sample_images()
            self.calculate_roc_auc()

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def add_noise(self, input: Tensor, **kwargs) -> Tensor:
        b, c, h, w = input.shape
        noise = self.noise * torch.randn((b, c, h, w))
        noise_img = input + noise.to(self.curr_device)
        return noise_img

    def mask(self, input: Tensor, **kwargs) -> Tensor:
        b, c, h, w = input.shape
        noise = np.random.choice([1, 0], size=(b,c,h,w), p=[1 - self.noise, self.noise])
        noise_img = input * torch.FloatTensor(noise).to(self.curr_device)
        return noise_img

    def configure_optimizers(self):
        vae_optimizer = torch.optim.Adam([{'params': self.Encoder.parameters()},
                                          {'params': self.Decoder.parameters()}], lr=self.params['LR1'])
        # vae_optimizer = torch.optim.Adam(self.Encoder.parameters(), lr=self.params['LR1'])
        g_optimizer = torch.optim.Adam(self.Decoder.parameters(), lr=self.params['LR1'])
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.params['LR2'])
        return [vae_optimizer, g_optimizer, d_optimizer], []

    def train_dataloader(self):
        dataloader, self.num_train_imgs = self.get_dataloader(train_split=True, abnormal_data=False, shuffle=True)

        return dataloader

    def val_dataloader(self):
        self.sample_dataloader, _ = self.get_dataloader(train_split=False, abnormal_data=False, shuffle=True)
        self.num_val_imgs = len(self.sample_dataloader)

        return self.sample_dataloader

    def get_dataloader(self, train_split: bool, abnormal_data: bool = False, shuffle: bool = True):
        transform = self.data_transforms()

        split = "train" if train_split else "val"

        if self.params['dataset'] == 'concrete-cracks':
            dataset = ConcreteCracksDataset(root_dir=self.params['data_path'],
                                            split=split,
                                            abnormal_data=abnormal_data,
                                            transform=transform)
        elif self.params['dataset'] == 'SDNET2018':
            dataset = SDNet2018(root_dir=self.params['data_path'],
                                split=split,
                                abnormal_data=abnormal_data,
                                transform=transform)
        else:
            raise ValueError('Undefined dataset type')

        dataloader = DataLoader(dataset,
                                batch_size=self.params['batch_size'],
                                shuffle=shuffle,
                                drop_last=True,
                                **self.additional_dataloader_args)

        return dataloader, len(dataset)

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        recons = self.generate(test_input, labels=test_label)

        self.logger.experiment.add_images("reconstructions",
                                          make_grid(recons, normalize=True, nrow=1),  # Use make_grid to normalize
                                          global_step=self.current_epoch,
                                          dataformats='CWH')  # make_grid seems to return channel x width x height

        try:
            samples = self.sample(144, self.curr_device, labels=test_label)
            self.logger.experiment.add_images("samples",
                                              make_grid(samples, normalize=True, nrow=12),
                                              global_step=self.current_epoch,
                                              dataformats='CWH')
        except:
            pass

        del test_input, recons  # , samples

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

        samples = self.decoder(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)

    def generate_noise(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        if self.noise_type == 'gaussian':
            noise_img = self.add_noise(x)
        elif self.noise_type == 'mask':
            noise_img = self.mask(x)

        return noise_img

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        if self.params['dataset'] == 'concrete-cracks':
            transform = transforms.Compose([transforms.Resize((self.params['img_size'], self.params['img_size'])),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'SDNET2018':
            transform = transforms.Compose([transforms.Resize((self.params['img_size'], self.params['img_size'])),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            raise ValueError('Undefined dataset type')
        return transform

    def calculate_pixelwise_differences(self, dataloader):
        all_labels = []
        all_pixelwise_differences = []

        for batch, labels in dataloader:
            # If model is on GPU moving the data to GPU as well is required, otherwise an error is thrown
            batch = batch.to(self.curr_device)

            predictions = self.generate(batch)

            pixelwise_difference = torch.mean(torch.abs(predictions - batch), dim=(1, 2, 3)).cpu()

            all_labels = np.concatenate([all_labels, labels])
            all_pixelwise_differences = np.concatenate([all_pixelwise_differences, pixelwise_difference], axis=0)

        return all_labels, all_pixelwise_differences

    def calculate_roc_auc(self):
        all_labels, all_pixelwise_differences = self.calculate_pixelwise_differences(self.sample_dataloader)

        val_dataloader_abnormal, _ = self.get_dataloader(train_split=False, abnormal_data=True, shuffle=True)
        _labels, _pixelwise_differences = self.calculate_pixelwise_differences(val_dataloader_abnormal)

        all_labels = np.concatenate([all_labels, _labels])
        all_pixelwise_differences = np.concatenate([all_pixelwise_differences, _pixelwise_differences])

        fpr, tpr, thresholds = roc_curve(all_labels, all_pixelwise_differences)
        roc_auc = auc(fpr, tpr)

        best_threshold = thresholds[np.argmax(tpr - fpr)]

        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        self.logger.experiment.add_figure("ROC Curve", plt.gcf(), global_step=self.current_epoch)



# class EncoderDecoder(nn.Module):
#     def __init__(self, in_channels, role,  ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """
#         Parameters:
#             in_channels (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the first conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(EncoderDecoder, self).__init__()
#         kw = 4
#         padw = 1
#         self.role = role
#         if role == "encoder":
#             sequence = [nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw), norm_layer(ndf), nn.ReLU()]
#             nf_mult = 1
#             nf_mult_prev = 1
#             for n in range(1, n_layers):  # gradually increase the number of filters
#                 nf_mult_prev = nf_mult
#                 nf_mult = min(2 ** n, 8)
#                 sequence += [
#                     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
#                     norm_layer(ndf * nf_mult),
#                     nn.ReLU()
#                 ]
#         else:
#             sequence = [nn.ConvTranspose2d(ndf, in_channels, kernel_size=kw, stride=2, padding=padw)]
#             nf_mult = 1
#             nf_mult_prev = 1
#             for n in range(1, n_layers):  # gradually increase the number of filters
#                 nf_mult_prev = nf_mult
#                 nf_mult = min(2 ** n, 8)
#                 sequence += [
#                     nn.ReLU(),
#                     norm_layer(ndf * nf_mult_prev),
#                     nn.ConvTranspose2d(ndf * nf_mult, ndf * nf_mult_prev, kernel_size=kw, stride=2, padding=padw, bias=False)
#                 ]
#             sequence = sequence[::-1]
#         self.blocks = nn.Sequential(*sequence)
#
#         if role == "encoder":
#             self.mean_head = self._build_head(ndf*nf_mult, norm_layer)
#             self.logvar_head = self._build_head(ndf*nf_mult, norm_layer)
#
#     @staticmethod
#     def _build_head(n_dim, norm_layer):
#         return nn.Sequential(
#             nn.Conv2d(n_dim, n_dim, 1),
#             norm_layer(n_dim),
#             nn.ReLU()
#             )
#
#     def forward(self, x):
#         x = self.blocks(x)
#         if self.role == "encoder":
#             return self.mean_head(x), self.logvar_head(x)
#         else:
#             return x

# class AnoVAE(nn.Module):
#     def __init__(self, in_channels, n_layers, ndf=64):
#         super(AnoVAE, self).__init__()
#         self.encoder = EncoderDecoder(in_channels, "encoder", ndf=ndf, n_layers=n_layers)
#         self.decoder = EncoderDecoder(in_channels, "decode", ndf=ndf, n_layers=n_layers)
#
#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         if self.training:
#             std = torch.exp(0.5*logvar)
#             eps = torch.randn_like(std)
#
#             z = mu + std * eps
#         else:
#             z = mu
#         logits = self.decoder(z)
#         out = {
#             'mu': mu,
#             'logvar': logvar,
#             'logits': logits,
#             'rec': torch.sigmoid(logits)
#         }
#         return out

# class Discriminator(nn.Module):
#     def __init__(self, in_channels, input_size, n_layers=3, ndf=64, norm_layer=nn.BatchNorm2d):
#         """
#         Parameters:
#             in_channels (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the first conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(Discriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#         self.conv_blocks = nn.Sequential(*sequence)
#         spatial_dim = ((input_size[0])*(input_size[1]))//(2**(2*n_layers))
#         self.fc = nn.Sequential(
#             nn.Linear(ndf*nf_mult*spatial_dim, 128),
#             nn.LeakyReLU(0.2, True),
#             nn.Linear(128, 1))
#
#     def forward(self, x):
#         x = self.conv_blocks(x)
#         x = x.view(x.shape[0], -1)
#         return self.fc(x)

# class AnoVAE(nn.Module):
#
#     def __init__(self,
#                  in_channels: int,
#                  latent_dim: int,
#                  hidden_dims: List = None,
#                  **kwargs) -> None:
#         super(AnoVAE, self).__init__()
#
#         self.latent_dim = latent_dim
#
#         modules = []
#         if hidden_dims is None:
#             hidden_dims = [32, 64, 128, 256, 256, 512]
#
#         _in_channels = in_channels
#
#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(_in_channels, out_channels=h_dim,
#                               kernel_size=3, stride=2, padding=1),
#                     nn.BatchNorm2d(h_dim),
#                     nn.LeakyReLU())
#             )
#             _in_channels = h_dim
#
#         self.encoder = nn.Sequential(*modules)
#         # TODO I hacked this hardcoded value into here because original author did the same, should be better done
#         self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
#         self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
#
#         # Build Decoder
#         modules = []
#
#         # TODO Same as above, redo the hardcoded stuff
#         self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
#
#         hidden_dims.reverse()
#
#         for i in range(len(hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(hidden_dims[i],
#                                        hidden_dims[i + 1],
#                                        kernel_size=3,
#                                        stride=2,
#                                        padding=1,
#                                        output_padding=1),
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     nn.LeakyReLU())
#             )
#
#         self.decoder = nn.Sequential(*modules)
#
#         self.final_layer = nn.Sequential(
#             nn.ConvTranspose2d(hidden_dims[-1],
#                                hidden_dims[-1],
#                                kernel_size=3,
#                                stride=2,
#                                padding=1,
#                                output_padding=1),
#             nn.BatchNorm2d(hidden_dims[-1]),
#             nn.LeakyReLU(),
#             nn.Conv2d(hidden_dims[-1], out_channels=3,
#                       kernel_size=3, padding=1),
#             nn.Tanh())
#
#         hidden_dims.reverse()
#         # self.save_hyperparameters()
#
#     def encode(self, input: Tensor) -> List[Tensor]:
#         """
#         Encodes the input by passing through the encoder network
#         and returns the latent codes.
#         :param input: (Tensor) Input tensor to encoder [N x C x H x W]
#         :return: (Tensor) List of latent codes
#         """
#         result = self.encoder(input)
#         result = torch.flatten(result, start_dim=1)
#
#         # Split the result into mu and var components
#         # of the latent Gaussian distribution
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)
#
#         return [mu, log_var]
#
#     def decode(self, z: Tensor) -> Tensor:
#         """
#         Maps the given latent codes
#         onto the image space.
#         :param z: (Tensor) [B x D]
#         :return: (Tensor) [B x C x H x W]
#         """
#         result = self.decoder_input(z)
#         # TODO These hardcoded values should also be changed
#         result = result.view(-1, 512, 2, 2)
#         result = self.decoder(result)
#         result = self.final_layer(result)
#         return result
#
#     def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
#         """
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu
#
#     def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
#         mu, log_var = self.encode(input)
#         z = self.reparameterize(mu, log_var)
#         return [self.decode(z), input, mu, log_var]

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256, 512]

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

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
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

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [z, x, mu, log_var]

class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256, 512]

        _in_channels = in_channels
        # Build Decoder

        # TODO Same as above, redo the hardcoded stuff
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        modules = []
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

    def decode(self, z: Tensor) -> Tensor:
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

    def forward(self, z: Tensor) -> Tensor:
        result = self.decode(z)
        return result



class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256, 512]

        _in_channels = in_channels

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(_in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            _in_channels = h_dim

        self.conv_blocks = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 4, latent_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(latent_dim, 1))

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)
