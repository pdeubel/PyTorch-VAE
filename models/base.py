from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_curve, auc
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA, MNIST

from datasets.test_gui_dataset import TestGUIDataset
from models.types_ import Any, List, Tensor


class BaseVAE(pl.LightningModule):

    def __init__(self, params: dict) -> None:
        super().__init__()

        self.params = params
        self.curr_device = None

        try:
            num_workers = params["dataloader_workers"]
        except KeyError:
            num_workers = 0

        self.additional_dataloader_args = {'num_workers': num_workers, 'pin_memory': True}

        # Used for visualizing
        self.denormalize = lambda x: (x + 1.0) / 2.0

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

        if self.current_epoch % 1 == 0 or self.current_epoch == (self.trainer.max_epochs - 1):
            self.sample_images()

        # if self.current_epoch == (self.trainer.max_epochs - 1):
        #     # We are in the last epoch, calculate ROC and AUC and log to tensorboard
        #     self.sample_images()
        #     self.calculate_roc_auc()

        self.logger.experiment.log({'avg_val_loss': avg_loss})

        return {'val_loss': avg_loss}

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

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)

        recons = self.generate(test_input)
        samples = self.sample(self.params["batch_size"], self.curr_device)

        self.logger.experiment.add_images("originals",
                                          self.denormalize(test_input),
                                          global_step=self.current_epoch)

        self.logger.experiment.add_images("reconstructions",
                                          self.denormalize(recons),
                                          global_step=self.current_epoch)

        self.logger.experiment.add_images("samples",
                                          self.denormalize(samples),
                                          global_step=self.current_epoch)

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
        dataloader, self.num_train_imgs = self.get_dataloader(train_split=True)

        return dataloader

    def val_dataloader(self):
        self.sample_dataloader, _ = self.get_dataloader(train_split=False)
        self.num_val_imgs = len(self.sample_dataloader)

        return self.sample_dataloader

    def get_dataloader(self, train_split: bool):
        transform = self.data_transforms()

        split = "train" if train_split else "val"
        shuffle = True if train_split else False

        if self.params['dataset'] == 'celeba':
            if not train_split:
                # CelebA uses no validation split, this comes from original author, but CelebA is not used in this
                # project anyway
                split = "test"
            dataset = CelebA(root=self.params['data_path'],
                             split=split,
                             transform=transform,
                             download=False)
        elif self.params["dataset"] == "test-gui":
            dataset = TestGUIDataset(root_dir=self.params["data_path"],
                                     split=split,
                                     transform=transform)
        elif self.params['dataset'] == 'MNIST':
            dataset = MNIST(root=self.params['data_path'],
                            train=train_split,
                            download=False,
                            transform=transform)
        else:
            raise ValueError('Undefined dataset type')

        dataloader = DataLoader(dataset,
                                batch_size=self.params['batch_size'],
                                shuffle=shuffle,
                                drop_last=False,
                                **self.additional_dataloader_args)

        return dataloader, len(dataset)

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize((self.params['img_size'], self.params['img_size'])),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params["dataset"] == "test-gui":
            transform = transforms.Compose([transforms.Resize((self.params['img_size'], self.params['img_size'])),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'MNIST':
            transform = transforms.Compose([transforms.ToTensor()])

            # Since we do not transform the input here set the denormalizing function to the identity function
            self.denormalize = lambda x: x
        else:
            raise ValueError('Undefined dataset type')
        return transform
