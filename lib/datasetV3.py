import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import Compose, RandomFlip_LR, RandomFlip_UD, RandomRotate
from .extract_patches import load_data


class TrainDatasetV3(Dataset):
    def split(self, imgs):
        ratio = 0.2

        val_idx = np.random.choice(len(imgs), replace=True, p=ratio)
        train_idx = np.delete(np.arange(len(imgs)), val_idx)

        # shuffle
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)

        return imgs[train_idx], imgs[val_idx]

    def __init__(self, data_path_list_file, mode, args):
        self.imgs, self.gts = load_data(data_path_list_file)
        self.mode = mode
        self.transforms = Compose(
            [RandomFlip_LR(prob=0.5), RandomFlip_UD(prob=0.5), RandomRotate()]
        )
        self.train, self.val = self.split(self.imgs)

    def __len__(self):
        if self.mode == "train":
            return len(self.train)
        else:
            return len(self.val)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.train[idx]
        else:
            img = self.val[idx]
        gt = self.gts[idx]
        if self.mode == "train":
            img, gt = self.transforms(img, gt)
        return img, gt
