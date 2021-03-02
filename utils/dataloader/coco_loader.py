import os
import collections
import json

import os.path as osp
import numpy as np
from PIL import Image
import collections
import torch
import torchvision

from torch.utils import data
from tqdm import tqdm
from torch.utils import data
import random 

from scipy.io import loadmat



class cocoloader(data.Dataset):
    def __init__(self,root='./datasets/coco/', split="multi-label-train2014", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.n_classes = 80
        self.img_transform = img_transform
        self.label_transform = label_transform
        if split == "test":
            with open(root+"test2014imgs.txt") as f:
                tmp = f.readlines()
            self.Imglist = [s.rstrip() for s in tmp]
        else:
            filePath = self.root + self.split + '.pkl'
            datafile = np.load(filePath, allow_pickle=True)
            self.GT = list(datafile.values())
            self.Imglist = list(datafile.keys())


    def __len__(self):
        return len(self.Imglist)

    def __getitem__(self, index):
        img = Image.open(self.Imglist[index]).convert('RGB')
        if self.split == "test":
            lbl_num = [-1, -1]
        else:
            lbl_num = self.GT[index]
        lbl = np.zeros(self.n_classes)
        if lbl_num[0] != -1:
            lbl[lbl_num] = 1


        seed = np.random.randint(2147483647)
        random.seed(seed)
        if self.img_transform is not None:
            img_o = self.img_transform(img)
            # img_h = self.img_transform(self.h_flip(img))
            # img_v = self.img_transform(self.v_flip(img))
            imgs = img_o
        else:
            imgs = img
        random.seed(seed)
        if self.label_transform is not None:
            label_o = self.label_transform(lbl)
            # label_h = self.label_transform(self.h_flip(label))
            # label_v = self.label_transform(self.v_flip(label))
            lbls = label_o
        else:
            lbls = lbl

 

        return imgs, lbls


