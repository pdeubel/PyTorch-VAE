import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ConcreteCracksDataset(Dataset):

    """
    Concrete Crack Images for Classification Dataset.

    http://dx.doi.org/10.17632/5y9wdsg2zt.2
    """

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        self.train_split = 0.8
        self.test_split = 0.2

        self.data_dir = os.path.join(self.root_dir, "Negative")
        self.train_index = round(len(os.listdir(self.data_dir)) * self.train_split)

        self.train_length = len(os.listdir(self.data_dir)[:self.train_index])
        self.test_length = len(os.listdir(self.data_dir)[self.train_index:])

    def __len__(self):
        if self.train:
            return self.train_length
        else:
            return self.test_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            image_path = os.path.join(self.data_dir, os.listdir(self.data_dir)[idx])
        else:
            image_path = os.path.join(self.data_dir, os.listdir(self.data_dir)[self.train_index + idx])

        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)

        label = 0  # TODO currently only negative samples are used, I don't think we need a label for standard VAE
                   #      training but this framework requires it

        return img, label
