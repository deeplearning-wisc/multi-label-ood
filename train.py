import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import validate
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data
from PIL import Image
from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.coco_loader import *
import random

import torchvision.transforms as transforms
#---- your own transformations
from utils.transform import ReLabel, ToLabel, ToSP, Scale

from model.classifiersimple import *
import torchvision

def train(args):
    args.save_dir += args.dataset + '/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((256),scale=(0.5, 2.0)),
            transforms.ToTensor(),
            normalize,
        ])

    label_transform = transforms.Compose([
            ToLabel(),
        ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == "pascal":
        loader = pascalVOCLoader(
                                 "./datasets/pascal/", 
                                 img_transform = img_transform, 
                                 label_transform = label_transform)
        val_data = pascalVOCLoader('./datasets/pascal/', split="voc12-val",
                                   img_transform=img_transform,
                                   label_transform=label_transform)
    elif args.dataset == "coco":
        loader = cocoloader("./datasets/coco/",
                            img_transform = img_transform,
                            label_transform = label_transform)
        val_data = cocoloader("./datasets/coco/", split="multi-label-val2014",
                            img_transform = val_transform,
                            label_transform = label_transform)
    elif args.dataset == "nus-wide":
        loader = nuswideloader("./datasets/nus-wide/",
                            img_transform = img_transform,
                            label_transform = label_transform)
        val_data = nuswideloader("./datasets/nus-wide/", split="val",
                                 img_transform=val_transform,
                                 label_transform=label_transform)
    else:
        raise AssertionError

    args.n_classes = loader.n_classes
    trainloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(val_data, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)

    print("number of images = ", len(loader))
    print("number of classes = ", args.n_classes, " architecture used = ", args.arch)

    if args.arch == "resnet101":
        orig_resnet = torchvision.models.resnet101(pretrained=True)
        features = list(orig_resnet.children())
        model = nn.Sequential(*features[0:8])
        clsfier = clssimp(2048, args.n_classes)
    elif args.arch == "densenet":
        orig_densenet = torchvision.models.densenet121(pretrained=True)
        features = list(orig_densenet.features)
        model = nn.Sequential(*features, nn.ReLU(inplace=True))
        clsfier = clssimp(1024, args.n_classes)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model).cuda()
        clsfier = nn.DataParallel(clsfier).cuda()

    if args.load == 1:
        model.load_state_dict(torch.load(args.save_dir + args.arch + ".pth"))
        clsfier.load_state_dict(torch.load(args.save_dir + args.arch +'clssegsimp' + ".pth"))
        print("Model loaded!")

    optimizer = torch.optim.Adam([{'params': model.parameters(),'lr':args.l_rate/10},{'params': clsfier.parameters()}], lr=args.l_rate)
    freeze_bn_affine = 1
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if freeze_bn_affine:
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    print("Saving to " + args.save_dir + args.arch + ".pth")
    bceloss = nn.BCEWithLogitsLoss()
    best_map = 0.0
    for epoch in range(args.n_epoch):
        model.train()
        clsifier.train()
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda().float())

            optimizer.zero_grad()
         
            outputs = model(images)
            outputs = clsfier(outputs)
            loss = bceloss(outputs, labels)

            loss.backward()
            optimizer.step()

        mAP = validate.validate(args, model, clsfier, val_loader)

        if mAP > best_map:
            torch.save(model.state_dict(), args.save_dir + args.arch + ".pth")
            torch.save(clsfier.state_dict(), args.save_dir + args.arch + 'clssegsimp' + ".pth")
            best_map = mAP
            print("Epoch [%d/%d][saved] Loss: %.4f mAP: %.4f" % (epoch + 1, args.n_epoch, loss.data, mAP))
        else:
            print("Epoch [%d/%d][----] Loss: %.4f mAP: %.4f" % (epoch + 1, args.n_epoch, loss.data, mAP))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet101') 
    parser.add_argument('--dataset', nargs='?', type=str, default='nus-wide',
                        help='Dataset to use [\'pascal, coco, nus-wide\']')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=30,
                        help='# of the epochs')
    parser.add_argument('--n_classes', nargs='?', type=int, default=20)
    parser.add_argument('--batch_size', nargs='?', type=int, default=320,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4,
                        help='Learning Rate')

    #save and oad
    parser.add_argument('--load', nargs='?', type=int)
    parser.add_argument('--save_dir', type=str, default="./savedmodels/")

    args = parser.parse_args()
    train(args)