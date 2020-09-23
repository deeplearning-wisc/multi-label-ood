import os
import os.path
import sys
import torch
import argparse
import torchvision
import lib
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
from utils.transform import ReLabel, ToLabel, ToSP, Scale
from model.classifiersimple import *
from PIL import Image
from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.coco_loader import *
import random
import torchvision.transforms as transforms
# ---- your own transformations
from utils.dataloader.folder import PlainDatasetFolder
from utils import anom_utils

def validate(args):
    print("In-dis data: "+args.dataset)
    print("Out-dis data: " + args.ood_data)
    torch.manual_seed(0)
    ###################### Setup Dataloader ######################
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize,
    ])
    label_transform = transforms.Compose([
        ToLabel(),
    ])
    # in_dis
    if args.dataset == 'pascal':
        train_data = pascalVOCLoader('./datasets/pascal/',
                                     img_transform=img_transform, label_transform=label_transform)
        test_data = pascalVOCLoader('./datasets/pascal/', split="voc12-test",
                                    img_transform=img_transform, label_transform=None)
        val_data = pascalVOCLoader('./datasets/pascal/', split="voc12-val",
                                   img_transform=img_transform, label_transform=label_transform)

    elif args.dataset == 'coco':
        train_data = cocoloader("./datasets/coco/",
                             img_transform = img_transform, label_transform = label_transform)
        val_data = cocoloader('./datasets/coco/', split="multi-label-val2014",
                            img_transform=img_transform, label_transform=label_transform)
        test_data = cocoloader('./datasets/coco/', split="test",
                               img_transform=img_transform, label_transform=None)

    elif args.dataset == "nus-wide":
        train_data = nuswideloader("./datasets/nus-wide/",
                            img_transform = img_transform, label_transform = label_transform)
        val_data = nuswideloader("./datasets/nus-wide/", split="val",
                            img_transform = img_transform, label_transform = label_transform)
        test_data = nuswideloader("./datasets/nus-wide/", split="test",
                            img_transform = img_transform, label_transform = label_transform)

    else:
        raise AssertionError

    args.n_classes = train_data.n_classes
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    in_test_loader = data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    # OOD data
    if args.ood_data == "imagenet":
        if args.dataset == "nus-wide":
            ood_root = "/nobackup-slow/dataset/nus-ood/"
            out_test_data = datasets.ImageFolder(ood_root, transform=img_transform)
        else:
            ood_root = "/nobackup-slow/dataset/ImageNet22k/ImageNet-22K"
            out_test_data = datasets.ImageFolder(ood_root, transform=img_transform)
    elif args.ood_data == "texture":
        ood_root = "/nobackup-slow/dataset/dtd/images/"
        out_test_data = datasets.ImageFolder(ood_root, transform = img_transform)
    elif args.ood_data == "MNIST":
        gray_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            normalize
        ])
        out_test_data = datasets.MNIST('/nobackup-slow/dataset/MNIST/',
                       train=False, transform=gray_transform)

    out_test_loader = data.DataLoader(out_test_data, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    ###################### Load Models ######################
    orig_resnet = torchvision.models.resnet101(pretrained=True)
    features = list(orig_resnet.children())
    model = nn.Sequential(*features[0:8])
    clsfier = clssimp(2048, args.n_classes)

    model = model.cuda()
    clsfier = clsfier.cuda()
    if torch.cuda.device_count() > 1:
        print("Using",torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        clsfier = nn.DataParallel(clsfier)

    model.load_state_dict(torch.load(args.load_model + args.arch + '.pth'))
    clsfier.load_state_dict(torch.load(args.load_model + args.arch + 'clssegsimp.pth'))
    print("model loaded!")

    # freeze the batchnorm and dropout layers
    model.eval()
    clsfier.eval()
    ###################### Compute Scores ######################
    if args.ood == "odin":
        print("Using temperature", args.T, "noise", args.noise)
        in_scores = lib.get_odin_scores(in_test_loader, model, clsfier, args.method,
                                        args.T, args.noise)
        out_scores = lib.get_odin_scores(out_test_loader, model, clsfier, args.method,
                                         args.T, args.noise)
    elif args.ood == "M":
        ## Feature Extraction
        temp_x = torch.rand(2, 3, 256, 256)
        temp_x = Variable(temp_x.cuda())
        temp_list = lib.model_feature_list(model, clsfier, temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        print('get sample mean and covariance')
        sample_mean, precision = lib.sample_estimator(model, clsfier, args.n_classes,
                                                      feature_list, train_loader)
        pack = (sample_mean, precision, num_output-1)
        print("Using noise", args.noise)
        in_scores = lib.get_Mahalanobis_score(model, clsfier, in_test_loader, pack,
                                              args.noise, args.n_classes)
        out_scores = lib.get_Mahalanobis_score(model, clsfier, out_test_loader, pack,
                                               args.noise, args.n_classes)

    else:
        in_scores = lib.get_logits(in_test_loader, model, clsfier, args, name="in_test")
        out_scores = lib.get_logits(out_test_loader, model, clsfier, args, name="out_test")

        if args.ood == "lof":
            val_scores = lib.get_logits(val_loader, model, clsfier, args, name="in_val")
            scores = lib.get_localoutlierfactor_scores(val_scores, in_scores, out_scores)
            in_scores = scores[:len(in_scores)]
            out_scores = scores[-len(out_scores):]

        if args.ood == "isol":
            val_scores = lib.get_logits(val_loader, model, clsfier, args, name="in_val")
            scores = lib.get_isolationforest_scores(val_scores, in_scores, out_scores)
            in_scores = scores[:len(in_scores)]
            out_scores = scores[-len(out_scores):]

    ###################### Measure ######################
    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    return auroc, aupr, fpr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    # ood measures
    parser.add_argument('--ood', type=str, default='logit',
                        help='which measure to use logit|odin|F|U|M')
    parser.add_argument('--method', type=str, default='max',
                        help='which method to use max|sum|cndsum')
    parser.add_argument('--average', type=int, default=0)
    # dataset
    parser.add_argument('--dataset', type=str, default='pascal')
    parser.add_argument('--ood_data', type=str, default='imagenet')
    parser.add_argument('--arch', type=str, default='resnet101',
                        help='Architecture to use')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch Size')
    parser.add_argument('--n_classes', type=int, default=20, help='# of classes')
    # save and load
    parser.add_argument('--save_path', type=str, default="./logits/", help="save the logits")
    parser.add_argument('--load_model', type=str, default="savedmodels/",
                        help="load model")
    # input pre-processing
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    args = parser.parse_args()
    args.load_model += args.dataset + '/'

    auprs = []
    aurocs=[]
    fprs=[]

    if args.average:
        sets = ["imagenet", "MNIST", "texture"]
    else:
        sets = [args.ood_data]

    save_path = args.save_path
    for test in sets:
        args.ood_data = test
        args.save_path = save_path + args.dataset + '/' + args.ood_data + '/'

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        auroc, aupr, fpr = validate(args)
        auprs.append(aupr)
        aurocs.append(auroc)
        fprs.append(fpr)

    anom_utils.print_measures(np.mean(aurocs), np.mean(auprs), np.mean(fprs), "Mean", "result")

