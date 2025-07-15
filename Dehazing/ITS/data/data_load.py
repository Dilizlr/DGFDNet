import os

import numpy as np
import torch
from data import (PairCompose, PairRandomCrop, PairRandomHorizontalFilp,
                  PairToTensor)
from PIL import Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    # image_dir = os.path.join(path, 'train')
    image_dir = path

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    root_path = os.path.join('/',*path.split('/')[:-2],'SOTS/')
    if path.split('/')[-2] == 'ITS':
        datatype = 'indoor/'
    else:
        datatype = 'outdoor/'
    image_dir = os.path.join(root_path, datatype)

    # image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    root_path = os.path.join('/',*path.split('/')[:-2],'SOTS/')
    if path.split('/')[-2] == 'ITS':
        datatype = 'indoor/'
    else:
        datatype = 'outdoor/'
    image_dir = os.path.join(root_path, datatype)
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx]))
        if self.is_valid or self.is_test:
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png'))
        else:
            label = Image.open(os.path.join(self.image_dir, 'clear', self.image_list[idx].split('_')[0]+'.png'))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
