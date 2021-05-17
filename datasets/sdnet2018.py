import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SDNet2018(Dataset):
    """

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

        self.data_directories_normal = [
            [os.path.join(self.root_dir, "D", "UD", _dir) for _dir in os.listdir(os.path.join(self.root_dir, "D", "UD"))],
            [os.path.join(self.root_dir, "P", "UP", _dir) for _dir in os.listdir(os.path.join(self.root_dir, "P", "UP"))],
            [os.path.join(self.root_dir, "W", "UW", _dir) for _dir in os.listdir(os.path.join(self.root_dir, "W", "UW"))]
        ]

        self.data_directories_abnormal = [
            [os.path.join(self.root_dir, "D", "CD", _dir) for _dir in os.listdir(os.path.join(self.root_dir, "D", "CD"))],
            [os.path.join(self.root_dir, "P", "CP", _dir) for _dir in os.listdir(os.path.join(self.root_dir, "P", "CP"))],
            [os.path.join(self.root_dir, "W", "CW", _dir) for _dir in os.listdir(os.path.join(self.root_dir, "W", "CW"))]
        ]

        # Make indices for each subdirectory of the dataset. Doing only one indice on all the subdirectories together
        # would result in a behavior that the testing dataset is always at the back (so for example the last x percent
        # But then all the testing would come from one category (i.e. walls, "W") and that is not desired
        self.train_indices_normal = [round(len(_dir) * self.train_split) for _dir in self.data_directories_normal]
        self.val_indices_normal = [round(len(_dir) * self.val_split) for _dir in self.data_directories_normal]
        self.test_indices_normal = [len(_dir) for _dir in self.data_directories_normal] - np.sum([self.train_indices_normal, self.val_indices_normal], axis=0)

        self.train_length_normal = sum(self.train_indices_normal)
        self.val_length_normal = sum(self.val_indices_normal)
        self.test_length_normal = sum([len(_dir) for _dir in self.data_directories_normal]) - self.train_length_normal - self.val_length_normal

        assert (self.train_length_normal + self.val_length_normal + self.test_length_normal ==
                sum(len(_dir) for _dir in self.data_directories_normal))

        # First create the index map for the normal (i.e. no cracks) training data
        # This index map will contain keys in range(0, self.train_length_normal) and tuples as values that first
        # choose which subdirectory of the data is sued and then which index in this subdirectory is used.
        # This is done for validation and testing sets and also for the abnormal data
        temp_index_map_1 = {i: (0, i) for i in range(self.train_indices_normal[0])}
        temp_index_map_2 = {self.train_indices_normal[0] + i: (1, i) for i in range(self.train_indices_normal[1])}
        temp_index_map_3 = {self.train_indices_normal[0] + self.train_indices_normal[1] + i: (2, i) for i in range(self.train_indices_normal[2])}

        self.index_map_train_normal = {**temp_index_map_1, **temp_index_map_2, **temp_index_map_3}

        temp_index_map_1 = {i: (0, self.train_indices_normal[0] + i) for i in range(self.val_indices_normal[0])}
        temp_index_map_2 = {self.val_indices_normal[0] + i: (1, self.train_indices_normal[1] + i) for i in range(self.val_indices_normal[1])}
        temp_index_map_3 = {self.val_indices_normal[0] + self.val_indices_normal[1] + i: (2, self.train_indices_normal[2] + i) for i in range(self.val_indices_normal[2])}

        self.index_map_val_normal = {**temp_index_map_1, **temp_index_map_2, **temp_index_map_3}

        temp_index_map_1 = {i: (0, self.train_indices_normal[0] + self.val_indices_normal[0] + i) for i in range(self.test_indices_normal[0])}
        temp_index_map_2 = {self.test_indices_normal[0] + i: (1, self.train_indices_normal[1] + self.val_indices_normal[1] + i) for i in range(self.test_indices_normal[1])}
        temp_index_map_3 = {self.test_indices_normal[0] + self.test_indices_normal[1] + i: (2, self.train_indices_normal[2] + self.val_indices_normal[2] + i) for i in range(self.test_indices_normal[2])}

        self.index_map_test_normal = {**temp_index_map_1, **temp_index_map_2, **temp_index_map_3}

        # Now create index maps for the abnormal data
        self.train_indices_abnormal = [round(len(_dir) * self.train_split) for _dir in self.data_directories_abnormal]
        self.val_indices_abnormal = [round(len(_dir) * self.val_split) for _dir in self.data_directories_abnormal]
        self.test_indices_abnormal = [len(_dir) for _dir in self.data_directories_abnormal] - np.sum([self.train_indices_abnormal, self.val_indices_abnormal], axis=0)

        self.train_length_abnormal = sum(self.train_indices_abnormal)
        self.val_length_abnormal = sum(self.val_indices_abnormal)
        self.test_length_abnormal = sum([len(_dir) for _dir in self.data_directories_abnormal]) - self.train_length_abnormal - self.val_length_abnormal

        assert (self.train_length_abnormal + self.val_length_abnormal + self.test_length_abnormal ==
                sum(len(_dir) for _dir in self.data_directories_abnormal))

        temp_index_map_1 = {i: (0, i) for i in range(self.train_indices_abnormal[0])}
        temp_index_map_2 = {self.train_indices_abnormal[0] + i: (1, i) for i in range(self.train_indices_abnormal[1])}
        temp_index_map_3 = {self.train_indices_abnormal[0] + self.train_indices_abnormal[1] + i: (2, i) for i in range(self.train_indices_abnormal[2])}

        self.index_map_train_abnormal = {**temp_index_map_1, **temp_index_map_2, **temp_index_map_3}

        temp_index_map_1 = {i: (0, self.train_indices_abnormal[0] + i) for i in range(self.val_indices_abnormal[0])}
        temp_index_map_2 = {self.val_indices_abnormal[0] + i: (1, self.train_indices_abnormal[1] + i) for i in range(self.val_indices_abnormal[1])}
        temp_index_map_3 = {self.val_indices_abnormal[0] + self.val_indices_abnormal[1] + i: (2, self.train_indices_abnormal[2] + i) for i in range(self.val_indices_abnormal[2])}

        self.index_map_val_abnormal = {**temp_index_map_1, **temp_index_map_2, **temp_index_map_3}

        temp_index_map_1 = {i: (0, self.train_indices_abnormal[0] + self.val_indices_abnormal[0] + i) for i in range(self.test_indices_abnormal[0])}
        temp_index_map_2 = {self.test_indices_abnormal[0] + i: (1, self.train_indices_abnormal[1] + self.val_indices_abnormal[1] + i) for i in range(self.test_indices_abnormal[1])}
        temp_index_map_3 = {self.test_indices_abnormal[0] + self.test_indices_abnormal[1] + i: (2, self.train_indices_abnormal[2] + self.val_indices_abnormal[2] + i) for i in range(self.test_indices_abnormal[2])}

        self.index_map_test_abnormal = {**temp_index_map_1, **temp_index_map_2, **temp_index_map_3}

    def __len__(self):
        if self.split == "train":
            if not self.abnormal_data:
                return self.train_length_normal
            else:
                return self.train_length_abnormal
        elif self.split == "val":
            if not self.abnormal_data:
                return self.val_length_normal
            else:
                return self.val_length_abnormal
        else:
            if not self.abnormal_data:
                return self.test_length_normal
            else:
                return self.test_length_abnormal

    def __getitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, list):
            raise RuntimeError("SDNet2018 Dataset was accessed with a list of indices, not sure if this works. Aborting")

        if self.split == "train":
            if not self.abnormal_data:
                label = 0
                index_map = self.index_map_train_normal
            else:
                label = 1
                index_map = self.index_map_train_abnormal
        elif self.split == "val":
            if not self.abnormal_data:
                label = 0
                index_map = self.index_map_val_normal
            else:
                label = 1
                index_map = self.index_map_val_abnormal
        else:
            if not self.abnormal_data:
                label = 0
                index_map = self.index_map_test_normal
            else:
                index_map = self.index_map_test_abnormal
                label = 1

        dataset_idx, idx_in_dataset = index_map[idx]

        if not self.abnormal_data:
            image_path = self.data_directories_normal[dataset_idx][idx_in_dataset]
        else:
            image_path = self.data_directories_abnormal[dataset_idx][idx_in_dataset]

        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)

        return img, label
