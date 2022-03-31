import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import Compose, RandomFlip_LR, RandomFlip_UD, RandomRotate
from .extract_patches import load_data


class TrainDatasetV3(Dataset):
    def split(self, ratio=0.2):
        self.val_idx = np.random.choice(
            len(self.imgs), size=int(len(self.imgs) * ratio), replace=True
        )
        self.train_idx = np.delete(np.arange(len(self.imgs)), self.val_idx)

        # shuffle
        np.random.shuffle(self.train_idx)
        np.random.shuffle(self.val_idx)

    def __init__(self, data_path_list_file, mode, args):
        self.imgs, self.gts = load_data(data_path_list_file)
        self.mode = mode
        self.transforms = Compose(
            [RandomFlip_LR(prob=0.5), RandomFlip_UD(prob=0.5), RandomRotate()]
        )
        self.split()

    def __len__(self):
        if self.mode == "train":
            return len(self.train_idx)
        else:
            return len(self.val_idx)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.imgs[self.train_idx[idx]]
            gt = self.gts[self.train_idx[idx]]
            img, gt = self.transforms(img, gt)
        else:
            img = self.imgs[self.val_idx[idx]]
            gt = self.gts[self.val_idx[idx]]
        return img / 255, gt / 255
