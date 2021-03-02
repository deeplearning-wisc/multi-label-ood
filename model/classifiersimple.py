import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

# GroupNorm

class clssimp(nn.Module):
    def __init__(self, ch=2880, num_classes=20):

        super(clssimp, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.way1 = nn.Sequential(
            nn.Linear(ch, 1000, bias=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
        )

        self.cls= nn.Linear(1000, num_classes,bias=True)

    def forward(self, x):
        # bp()
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        logits = self.cls(x)
        return logits

    def intermediate_forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        return x




class segclssimp_group(nn.Module):
    def __init__(self, ch=2880, num_classes=21):

        super(segclssimp_group, self).__init__()
        self.depthway1 = nn.Sequential(
            nn.Conv2d(ch, 1000, kernel_size=1),
            nn.GroupNorm(4,1000),
            nn.ReLU(inplace=True),
        )
        self.depthway2 = nn.Sequential(
            nn.Conv2d(1000, 1000, kernel_size=1),
            nn.GroupNorm(4,1000),
            nn.ReLU(inplace=True),
        )
        self.depthway3 = nn.Sequential(
            nn.Conv2d(1000, 512, kernel_size=1),
            nn.GroupNorm(4,512),
            nn.ReLU(inplace=True),
        )

        self.clsdepth = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # bp()

        seg = self.depthway1(x)
        seg = self.depthway2(seg)
        seg = self.depthway3(seg)
        seg = self.clsdepth(seg)



        return seg
