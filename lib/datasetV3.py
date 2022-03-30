import torch
from torch.utils.data import Dataset
from PIL import Image
from .extract_patches import load_data
from .dataset import Compose, RandomFlip_LR, RandomFlip_UD, RandomRotate


class TrainDatasetV3(Dataset):
    def __init__(self, data_path_list_file, mode, args):
        self.imgs, self.gts = load_data(data_path_list_file)

        self.transforms = None
        if mode == "train":
            self.transforms = Compose(
                [RandomFlip_LR(prob=0.5), RandomFlip_UD(prob=0.5), RandomRotate()]
            )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        gt = self.gts[idx]
        if self.transforms:
            img, gt = self.transforms(img, gt)
        return img, gt
