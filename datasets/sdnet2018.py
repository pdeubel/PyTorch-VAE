import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class SDNet2018(Dataset):
    """
    SDNet2018 dataset which contains images of cracked and non-cracked concrete bridge decks, walls and pavements.

    Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). SDNET2018: A concrete crack image dataset for machine learning
     applications. Utah State University. https://doi.org/10.15142/T3TD19
    """

    def __init__(self, root_dir, split: str = "train", abnormal_data: bool = False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, it assumes 'D', 'P' and 'W' as subfolders and these in
                turn contain subfolders with cracked and non-cracked images
            split (string): 'train', 'val' or 'test'
            abnormal_data (bool, optional): If this is true, the abnormal data is returned, i.e. images with cracks.
                Otherwise non-cracked images are returned.
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

        self.data_directories_normal = [
            [os.path.join(self.root_dir, "D", "UD", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "D", "UD"))],
            [os.path.join(self.root_dir, "P", "UP", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "P", "UP"))],
            [os.path.join(self.root_dir, "W", "UW", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "W", "UW"))]
        ]

        self.data_directories_abnormal = [
            [os.path.join(self.root_dir, "D", "CD", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "D", "CD"))],
            [os.path.join(self.root_dir, "P", "CP", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "P", "CP"))],
            [os.path.join(self.root_dir, "W", "CW", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "W", "CW"))]
        ]

        self.train_data_normal, self.val_data_normal, self.test_data_normal = self.create_splits(
            self.data_directories_normal)

        self.train_data_abnormal, self.val_data_abnormal, self.test_data_abnormal = self.create_splits(
            self.data_directories_abnormal)

        self.train_data_normal_length = len(self.train_data_normal)
        self.val_data_normal_length = len(self.val_data_normal)
        self.test_data_normal_length = len(self.test_data_normal)

        self.train_data_abnormal_length = len(self.train_data_abnormal)
        self.val_data_abnormal_length = len(self.val_data_abnormal)
        self.test_data_abnormal_length = len(self.test_data_abnormal)

    def create_splits(self, data_directories: list):
        """
        Create train, val and test splits from the provided data_directory. This expects that data_directories is a list
        of subfolders and each subfolder contains direct paths to the image files. This way the train, val and test
        splits can be done per subfolder.

        Doing splits on all subfolders concatenated would result in the test set containing always the last portion of
        the concatenated list. This is not desired as then it would contain images of only one 'class' (i.e. bridge
        decks, walls or pavements). Also the other splits would potentially not see data from this 'class'.
        """
        train_split = []
        val_split = []
        test_split = []

        for sub_dir in data_directories:
            train_index = round(len(sub_dir) * self.train_split)
            val_index = round(len(sub_dir) * self.val_split)

            train_split += sub_dir[:train_index]
            val_split += sub_dir[train_index:train_index + val_index]
            test_split += sub_dir[train_index + val_index:]

        return train_split, val_split, test_split

    def __len__(self):
        if not self.abnormal_data:
            if self.split == "train":
                return self.train_data_normal_length
            elif self.split == "val":
                return self.val_data_normal_length
            else:
                return self.test_data_normal_length
        else:
            if self.split == "train":
                return self.train_data_abnormal_length
            elif self.split == "val":
                return self.val_data_abnormal_length
            else:
                return self.test_data_abnormal_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, list):
            raise RuntimeError(
                "SDNet2018 Dataset was accessed with a list of indices, not sure if this works. Aborting")

        if not self.abnormal_data:
            label = 0
            if self.split == "train":
                data = self.train_data_normal
            elif self.split == "val":
                data = self.val_data_normal
            else:
                data = self.test_data_normal
        else:
            label = 1
            if self.split == "train":
                data = self.train_data_abnormal
            elif self.split == "val":
                data = self.val_data_abnormal
            else:
                data = self.test_data_abnormal

        img = Image.open(data[idx])

        if self.transform:
            img = self.transform(img)

        return img, label
