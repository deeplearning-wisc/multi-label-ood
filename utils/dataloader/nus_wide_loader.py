import os
import collections
import json

import numpy as np
from PIL import Image
import torch
import torchvision

from tqdm import tqdm
from torch.utils import data
import random



class nuswideloader(data.Dataset):
    def __init__(self, root='./datasets/nus-wide/', split="train",
                 in_dis=True, img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.in_dis = in_dis
        self.n_classes = 81
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.GT = []
        self.Imglist = []
        self.processing()
        if split != "train":
            self.partition()

    def processing(self):
        if self.split == "train":
            file_img = "nus_wide_train_imglist.txt"
            file_label = "nus_wide_train_label.txt"
        else:
            # validation and test
            file_img = "nus_wide_test_imglist.txt"
            file_label = "nus_wide_test_label.txt"
        f1 = open(self.root + file_img)
        img = f1.readlines()
        lbl = np.loadtxt(self.root + file_label, dtype=np.int64)
        # if self.in_dis:
        select = np.where(np.sum(lbl, axis=1) > 0)[0]
        # else:
        #     select = np.where(np.sum(lbl, axis=1) == 0)[0]

        self.GT = lbl[select]
        # img = [img[i].split()[0] for i in range(len(img))]
        # self.Imglist = [img[i].replace('\\','/') for i in select]
        self.Imglist = [img[i].split()[0] for i in select]

    def partition(self):
        np.random.seed(999)
        state = np.random.get_state()
        labels = self.GT
        imgs = self.Imglist
        num = labels.shape[0] // 2
        # num = labels.shape[0]
        np.random.shuffle(labels)
        np.random.set_state(state)
        np.random.shuffle(imgs)
        if self.split == "val":
            self.GT = labels[:num]
            self.Imglist = imgs[:num]
        else:
            self.GT = labels[num:]
            self.Imglist = imgs[num:]

    def __len__(self):
        return len(self.Imglist)

    def __getitem__(self, index):
        path = "/nobackup-slow/dataset/nus-wide/"
        img = Image.open(path + self.Imglist[index]).convert('RGB')
        if self.split == "test":
            lbl = -np.ones(self.n_classes)
        else:
            lbl = self.GT[index]


        if self.img_transform is not None:
            img_o = self.img_transform(img)
            # img_h = self.img_transform(self.h_flip(img))
            # img_v = self.img_transform(self.v_flip(img))
            imgs = img_o
        else:
            imgs = img
        # random.seed(seed)
        if self.label_transform is not None:
            label_o = self.label_transform(lbl)
            # label_h = self.label_transform(self.h_flip(label))
            # label_v = self.label_transform(self.v_flip(label))
            lbls = label_o
        else:
            lbls = lbl

        return imgs, lbls


