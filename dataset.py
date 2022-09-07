# -*- coding:utf-8 -*-
# @Time     : 2022/5/25  上午10:32
# @Author   : Rui Ma
# @Email    : ruima@std.uestc.edu.cn
# @File     : dataset.py
# @Software : PyCharm

import os
import torch
import collections
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Transforms
train_transforms = transforms.Compose([
    transforms.FiveCrop(256),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])
test_transforms = transforms.Compose([
    transforms.FiveCrop(256),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def image_score(gt_file):
    """Add image and score to table."""
    with open(gt_file) as f:
        lines = f.readlines()

    table = []
    for line in lines:
        score = float(line.split(' ')[0])
        img_name = line.split(' ')[1][:-1]
        table.append([img_name, score])

    return np.array(table)


def score_cal(gt_file):
    """Find max and min scores."""
    with open(gt_file) as f:
        lines = f.readlines()

    score_all = []
    for line in lines:
        score = float(line.split(' ')[0])
        score_all.append(score)

    score_all = np.array(score_all)
    score_min = np.min(score_all)
    score_max = np.max(score_all)

    return score_min, score_max


class Datasets(data.Dataset):
    """Datasets processing."""

    def __init__(self, root, txt_root, transform=None, train_phase=True):
        self.root = root
        self.train_root = root + 'train/'
        self.test_root = root + 'test/'
        self.txt_root = txt_root
        self.train_txt_root = self.train_root + 'train.txt'
        self.test_txt_root = self.test_root + 'test.txt'
        self.transform = transform
        self.train_phase = train_phase
        self.files = collections.defaultdict()
        self.score_min, self.score_max = score_cal(txt_root)

        train_table = image_score(self.train_txt_root)
        test_table = image_score(self.test_txt_root)

        if train_phase:
            self.phase = 'train'
            self.files[self.phase] = train_table
        else:
            self.phase = 'test'
            self.files[self.phase] = test_table

    def __getitem__(self, index):
        img_info = self.files[self.phase][index]

        # Load image
        if self.train_phase:
            img = Image.open(self.train_root + img_info[0])
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = Image.open(self.test_root + img_info[0])
            if self.transform is not None:
                img = self.transform(img)

        # Generate score
        t_score = float(img_info[1])
        t_score = (t_score - self.score_min) / (self.score_max - self.score_min)
        score = t_score

        return img, score

    def __len__(self):
        length = len(self.files[self.phase])
        return length


def train_loader(root, txt_root, batch_size, shuffle):
    """Train loader."""
    return data.DataLoader(
        Datasets(root=root, txt_root=txt_root, transform=train_transforms, train_phase=True),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True)


def test_loader(root, txt_root, batch_size, shuffle):
    """Test loader."""
    return data.DataLoader(
        Datasets(root=root, txt_root=txt_root, transform=test_transforms, train_phase=False),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True)


if __name__ == '__main__':
    options = {
        'dataset': 'LIEQ_Seed0',
        'batch_size': 5,
        'img_root': '../Data/SelectedDataset/',
    }

    img_root = options['img_root'] + options['dataset'] + '/'
    score_root = img_root + options['dataset'] + '.txt'
    train_data_loader = train_loader(root=img_root, txt_root=score_root, batch_size=options['batch_size'], shuffle=True)
    test_data_loader = test_loader(root=img_root, txt_root=score_root, batch_size=options['batch_size'], shuffle=False)
