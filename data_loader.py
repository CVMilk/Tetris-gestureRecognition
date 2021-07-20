# -*- coding: utf-8 -*-
"""
Created on 2021-07-18 

@File : data_loader.py

@Author : Liulei (mrliu_9936@163.com)

@Purpose : 定义自己的一个数据集

"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, image_label, mode='train'):
        self.image_label = image_label
        self.mode = mode

        if self.mode == 'train':
            # 数据增强
            self.train_augs = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomRotation(15),
                transforms.ToTensor()
            ])

        if self.mode == 'test':
            self.test_augs = transforms.Compose([
                transforms.ToTensor()
            ])


    def __getitem__(self, item):

        feature, label = self.image_label[item]
        feature = Image.open(feature).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        if self.mode == 'train':
            feature = self.train_augs(feature)
        else:
            feature = self.test_augs(feature)
        return feature, label


    def __len__(self):
        return len(self.image_label)