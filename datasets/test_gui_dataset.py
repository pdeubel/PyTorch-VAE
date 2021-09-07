import os

import torch
from PIL.Image import Image
from torch.utils.data import Dataset


class TestGUIDataset(Dataset):

    def __init__(self, root_dir, split: str = "train", transform=None):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.train_split = 0.65
        self.val_split = 0.15
        self.test_split = 0.2

        possible_splits = [
            "train",
            "val",
            "test"
        ]

        assert split in possible_splits, "Chosen split '{}' is not valid".format(split)

        self.train_index = round(len(os.listdir(self.root_dir)) * self.train_split)
        self.val_index = round(len(os.listdir(self.root_dir)) * self.val_split)

        self.train_length = len(os.listdir(self.root_dir))[:self.train_index]
        self.val_length = len(os.listdir(self.root_dir)[self.train_index:self.train_index + self.val_index])
        self.test_length = len(os.listdir(self.root_dir)[self.train_index + self.val_index:])

    def __len__(self):
        if self.split == "train":
            return self.train_length
        elif self.split == "val":
            return self.val_length
        else:
            return self.test_length

    def __getitem__(self, index):
        if torch.is_tensor(index):
            raise RuntimeError("Index is a tensor, not implementd in TestGUIDataset")

        if self.split == "train":
            image_path = os.path.join(self.root_dir, os.listdir(self.root_dir)[index])
        elif self.split == "val":
            image_path = os.path.join(self.root_dir, os.listdir(self.root_dir)[self.train_index + index])
        else:
            image_path = os.path.join(
                self.root_dir,
                os.listdir(self.root_dir)[self.train_index + self.val_index + index]
            )

        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)

        # None because we have no labels
        return img, None
