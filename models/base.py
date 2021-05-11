from abc import abstractmethod

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA

from datasets.concrete_cracks import ConcreteCracksDataset
from models.types_ import *


class BaseVAE(pl.LightningModule):

    def __init__(self, params: dict) -> None:
        super().__init__()

        self.params = params
        self.curr_device = None

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.loss_function(*results,
                                        M_N=self.params['batch_size'] / self.num_train_imgs,
                                        optimizer_idx=optimizer_idx,
                                        batch_idx=batch_idx)

        # TODO this is deprecated
        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.loss_function(*results,
                                      M_N=self.params['batch_size'] / self.num_val_imgs,
                                      optimizer_idx=optimizer_idx,
                                      batch_idx=batch_idx)

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}

        if self.current_epoch % 5 == 0 or self.current_epoch == (self.trainer.max_epochs - 1):
            self.sample_images()

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self, save=True, display=False):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.generate(test_input, labels=test_label)

        if save:
            vutils.save_image(recons.data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"recons_{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)

        if display:
            plt.imshow(vutils.make_grid(recons.data, normalize=True, nrow=12).permute(2, 1, 0).numpy())
            plt.title("Reconstructed images")
            plt.show()

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.sample(144,
                                  self.curr_device,
                                  labels=test_label)
            if save:
                vutils.save_image(samples.cpu().data,
                                  f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                                  f"{self.logger.name}_{self.current_epoch}.png",
                                  normalize=True,
                                  nrow=12)
            if display:
                plt.imshow(vutils.make_grid(samples.data, normalize=True, nrow=12).permute(2, 1, 0).numpy())
                plt.title("Sampled images")
                plt.show()
        except:
            pass

        del test_input, recons  # , samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(self.params['submodel'].parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root=self.params['data_path'],
                             split="train",
                             transform=transform,
                             download=False)
        elif self.params['dataset'] == "concrete-cracks":
            dataset = ConcreteCracksDataset(root_dir=self.params['data_path'],
                                            split="train",
                                            abnormal_data=False,
                                            transform=transform)
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            self.sample_dataloader = DataLoader(CelebA(root=self.params['data_path'],
                                                       split="test",
                                                       transform=transform,
                                                       download=False),
                                                batch_size=144,
                                                shuffle=True,
                                                drop_last=True)
        elif self.params['dataset'] == 'concrete-cracks':
            dataset = ConcreteCracksDataset(root_dir=self.params['data_path'],
                                            split="val",
                                            abnormal_data=False,
                                            transform=transform)
            self.sample_dataloader = DataLoader(dataset,
                                                batch_size=144,
                                                shuffle=True,
                                                drop_last=True)
        else:
            raise ValueError('Undefined dataset type')

        self.num_val_imgs = len(self.sample_dataloader)

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize((self.params['img_size'], self.params['img_size'])),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'concrete-cracks':
            transform = transforms.Compose([transforms.Resize((self.params['img_size'], self.params['img_size'])),
                                            transforms.ToTensor(),
                                            SetRange])
        else:
            raise ValueError('Undefined dataset type')
        return transform
