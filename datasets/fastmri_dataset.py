import os
import numpy as np
import torch
from torch.utils.data import Dataset


class FastMRIDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train"
    ):

        self.lr_dir = os.path.join(
            root_dir,
            split,
            "lr_kspace"
        )

        self.hr_dir = os.path.join(
            root_dir,
            split,
            "hr_image"
        )

        self.files = sorted(
            os.listdir(self.lr_dir)
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        lr_path = os.path.join(
            self.lr_dir,
            filename
        )

        hr_path = os.path.join(
            self.hr_dir,
            filename
        )

        lr = np.load(
            lr_path
        ).astype(np.float32)

        hr = np.load(
            hr_path
        ).astype(np.float32)

        lr = torch.from_numpy(lr)
        hr = torch.from_numpy(hr)

        return {
            "kspace": lr,
            "target": hr
        }