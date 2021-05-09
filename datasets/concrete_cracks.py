import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ConcreteCracksDataset(Dataset):
    """
    Concrete Crack Images for Classification Dataset.

    http://dx.doi.org/10.17632/5y9wdsg2zt.2
    """

    def __init__(self, root_dir, split: str = "train", abnormal_data: bool = False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.train_split = 0.65
        self.val_split = 0.15
        self.test_split = 0.2

        self.possible_splits = [
            "train",
            "val",
            "test"
        ]

        assert split in self.possible_splits, "Chosen split '{}' is not valid".format(split)

        self.split = split
        self.abnormal_data = abnormal_data

        self.data_dir_normal = os.path.join(self.root_dir, "Negative")
        self.data_dir_abnormal = os.path.join(self.root_dir, "Positive")

        # Use same splits for both normal and abnormal training
        assert len(os.listdir(self.data_dir_normal)) == len(os.listdir(self.data_dir_abnormal))

        self.train_index = round(len(os.listdir(self.data_dir_normal)) * self.train_split)
        self.val_index = round(len(os.listdir(self.data_dir_normal)) * (self.train_split + self.val_split))

        self.train_length = len(os.listdir(self.data_dir_normal)[:self.train_index])
        self.val_length = len(os.listdir(self.data_dir_normal)[self.train_index:self.train_index + self.val_index])
        self.test_length = len(os.listdir(self.data_dir_normal)[self.train_index + self.val_index:])

    def __len__(self):
        if self.split == "train":
            return self.train_length
        elif self.split == "val":
            return self.val_length
        else:
            return self.test_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.abnormal_data:
            data_dir = self.data_dir_normal
            label = 0
        else:
            data_dir = self.data_dir_abnormal
            label = 1

        if self.split == "train":
            image_path = os.path.join(data_dir, os.listdir(data_dir)[idx])
        elif self.split == "val":
            image_path = os.path.join(data_dir, os.listdir(data_dir)[self.train_index + idx])
        else:
            image_path = os.path.join(data_dir, os.listdir(data_dir)[self.train_index + self.val_index + idx])

        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)

        return img, label
