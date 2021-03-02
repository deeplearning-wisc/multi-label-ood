import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


from pdb import set_trace as bp

class mydensenet201_core(nn.Module):
    def __init__(self, pretrained=True):
        super(mydensenet201_core, self).__init__()
        densenet201 = models.densenet201(pretrained=True)

        features = list(densenet201.features.children())


        ch = 2880 

        self.conv1 = nn.Sequential(*features[:4])
        self.denseblock1 = nn.Sequential(*features[4])
        self.denseblock2 = nn.Sequential(*features[5:7])
        self.denseblock3 = nn.Sequential(*features[7:9])
        self.denseblock4 = nn.Sequential(*features[9:11])




    def forward(self, x):
        conv1 = self.conv1(x)
        denseblock1 = self.denseblock1(conv1)
        denseblock2 = self.denseblock2(denseblock1)
        denseblock3 = self.denseblock3(denseblock2)
        denseblock4 = self.denseblock4(denseblock3)

       
        x1 = conv1
        x2 = denseblock1
        x3 = F.upsample_bilinear(denseblock2, x1.size()[2:])
        x4 = F.upsample_bilinear(denseblock3, x1.size()[2:])
        x5 = F.upsample_bilinear(denseblock4, x1.size()[2:])
        xglobal = F.max_pool2d(x5, kernel_size=x5.size()[2:])
        # bp()
        out = torch.cat((x1, x2, x3, x4, x5,xglobal.expand(x1.size()[0],1920,x1.size()[2],x1.size()[3])), 1)
        # out = torch.cat((x1, x2, x3, x4, x5), 1)



        return out


class mydensenet121_core(nn.Module):
    def __init__(self, pretrained=True):
        super(mydensenet121_core, self).__init__()
        densenet121 = models.densenet121(pretrained=True)

        features = list(densenet121.features.children())


        ch = 2880 

        self.conv1 = nn.Sequential(*features[:4])
        self.denseblock1 = nn.Sequential(*features[4])
        self.denseblock2 = nn.Sequential(*features[5:7])
        self.denseblock3 = nn.Sequential(*features[7:9])
        self.denseblock4 = nn.Sequential(*features[9:11])




    def forward(self, x):
        conv1 = self.conv1(x)
        denseblock1 = self.denseblock1(conv1)
        denseblock2 = self.denseblock2(denseblock1)
        denseblock3 = self.denseblock3(denseblock2)
        denseblock4 = self.denseblock4(denseblock3)

       
        x1 = conv1
        x2 = denseblock1
        x3 = F.upsample_bilinear(denseblock2, x1.size()[2:])
        x4 = F.upsample_bilinear(denseblock3, x1.size()[2:])
        x5 = F.upsample_bilinear(denseblock4, x1.size()[2:])
        xglobal = F.max_pool2d(x5, kernel_size=x5.size()[2:])
        out = torch.cat((x1, x2, x3, x4, x5,xglobal.expand(x1.size()[0],1920,x1.size()[2],x1.size()[3])), 1)
        # out = torch.cat((x1, x2, x3, x4, x5), 1)



        return out