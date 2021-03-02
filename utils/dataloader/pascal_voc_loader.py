import os
import collections
import json

import numpy as np
from PIL import Image
import collections
import torch
import torchvision

from torch.utils import data
import random 

from scipy.io import loadmat


class pascalVOCLoader(data.Dataset):
    def __init__(self,root='./datasets/pascal/', split="voc12-train", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.n_classes = 20
        self.img_transform = img_transform
        self.label_transform = label_transform
        filePath = self.root + self.split + '.mat'
        datafile = loadmat(filePath)
        if split == "voc12-test":
            self.GT = None
        else:
            self.GT = datafile['labels']
        self.Imglist = datafile['Imlist']
        


    def __len__(self):
        return len(self.Imglist)

    def __getitem__(self, index):
        img = Image.open(self.Imglist[index].strip()).convert('RGB')
        if self.GT is not None:
            lbl = self.GT[index]
        else:
            lbl = -1

        seed = np.random.randint(2147483647)
        random.seed(seed)
        if self.img_transform is not None:
            img_o = self.img_transform(img)
            imgs = img_o
        else:
            imgs = img
        random.seed(seed)
        if self.label_transform is not None:
            label_o = self.label_transform(lbl)
            lbls = label_o
        else:
            lbls = lbl

        return imgs, lbls

