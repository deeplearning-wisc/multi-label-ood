import os
import os.path
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dataset
from torch.utils import data
from utils.transform import ReLabel, ToLabel, ToSP, Scale
from model.classifiersimple import *
from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.coco_loader import *
import lib
import random
import torchvision.transforms as transforms
from PIL import Image as PILImage
# ---- your own transformations
from utils.dataloader.folder import PlainDatasetFolder
from sklearn import metrics
from utils import anom_utils
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

def validate(args):
    # compute in_data score
    torch.manual_seed(0)
    np.random.seed(0)
    if args.ood == 'M':
        pack = (sample_mean, precision, num_output - 1)
        print('get Mahalanobis scores')
        in_scores = lib.get_Mahalanobis_score(model, clsfier, val_loader, pack,
                                              args.noise, args.n_classes)
    else:
        in_scores = lib.get_odin_scores(val_loader, model, clsfier, args.method,
                                        args.T, args.noise)

    ############################### ood ###############################
    ood_num_examples = len(val_data) // 5
    auroc_list = []
    aupr_list = []
    fpr_list = []
    # /////////////// Gaussion Noise ///////////////
    print("Gaussion noise detection")
    dummy_targets = -torch.ones(ood_num_examples, args.n_classes)
    ood_data = torch.from_numpy(np.float32(np.clip(
        np.random.normal(size=(ood_num_examples, 3, 256, 256), scale=0.5), -1, 1)))
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.n_workers)
    if args.ood == "M":
        out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
                                              args.noise, args.n_classes)

    else:
        out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
                                         args.T, args.noise)

    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    # /////////////// Uniform Noise ///////////////
    print('Uniform[-1,1] Noise Detection')
    dummy_targets = -torch.ones(ood_num_examples, args.n_classes)
    ood_data = torch.from_numpy(
        np.random.uniform(size=(ood_num_examples, 3, 256, 256),
                          low=-1.0, high=1.0).astype(np.float32))
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.n_workers)

    if args.ood == "M":
        out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
                                               args.noise, args.n_classes)
    else:
        out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
                                         args.T, args.noise)
    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    # #/////////////// Arithmetic Mean of Images ///////////////
    class AvgOfPair(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.shuffle_indices = np.arange(len(dataset))
            np.random.shuffle(self.shuffle_indices)

        def __getitem__(self, i):
            random_idx = np.random.choice(len(self.dataset))
            while random_idx == i:
                random_idx = np.random.choice(len(self.dataset))

            return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0

        def __len__(self):
            return len(self.dataset)

    ood_loader = torch.utils.data.DataLoader(AvgOfPair(val_data),
                                             batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.n_workers, pin_memory=True)

    print('\nArithmetic_Mean Detection')
    if args.ood == "M":
        out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
                                               args.noise, args.n_classes)
    else:
        out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
                                         args.T, args.noise)
    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    # # /////////////// Geometric Mean of Images ///////////////
    if args.dataset == 'pascal':
        modified_data = pascalVOCLoader('./datasets/pascal/', split="voc12-val",
                                   img_transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]))
    elif args.dataset == 'coco':
        modified_data = cocoloader('./datasets/coco/', split="multi-label-val2014",
                              img_transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]))
    elif args.dataset == "nus-wide":
        modified_data = nuswideloader("./datasets/nus-wide/", split="val",
                                 img_transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()]))

    modified_data = data.Subset(modified_data, indices)
    class GeomMeanOfPair(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.shuffle_indices = np.arange(len(dataset))
            np.random.shuffle(self.shuffle_indices)

        def __getitem__(self, i):
            random_idx = np.random.choice(len(self.dataset))
            while random_idx == i:
                random_idx = np.random.choice(len(self.dataset))

            return normalize(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0

        def __len__(self):
            return len(self.dataset)

    ood_loader = torch.utils.data.DataLoader(
        GeomMeanOfPair(modified_data), batch_size=args.batch_size, shuffle=True,
        num_workers=args.n_workers, pin_memory=True)

    print('\nGeometric_Mean Detection')
    if args.ood == "M":
        out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
                                               args.noise, args.n_classes)
    else:
        out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
                                         args.T, args.noise)
    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    # /////////////// Jigsaw Images ///////////////

    ood_loader = torch.utils.data.DataLoader(modified_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.n_workers, pin_memory=True)

    jigsaw = lambda x: torch.cat((
        torch.cat((torch.cat((x[:, 64:128, :128], x[:, :64, :128]), 1),
                   x[:, 128:, :128]), 2),
        torch.cat((x[:, 128:, 128:],
                   torch.cat((x[:, :128, 192:], x[:, :128, 128:192]), 2)), 2),
    ), 1)
    ood_loader.dataset.transform = transforms.Compose([transforms.Resize((256,256)),
        transforms.ToTensor(),jigsaw, normalize])

    print('\nJigsaw Detection')
    if args.ood == "M":
        out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
                                               args.noise, args.n_classes)
    else:
        out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
                                         args.T, args.noise)
    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)

    # /////////////// Mean Results ///////////////

    print('\n\nMean Validation Results')
    anom_utils.print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list),
                              ood="Mean", method="validation")
    return np.mean(fpr_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    # ood measures
    parser.add_argument('--ood', type=str, default='odin',
                        help='which measure to use odin|M')
    parser.add_argument('--method', type=str, default='max',
                        help='which method to use max|sum|cndsum')
    parser.add_argument('--dataset', type=str, default='pascal')
    parser.add_argument('--arch', type=str, default='resnet101',
                        help='Architecture to use')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch Size')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--n_classes', type=int, default=20)
    # save and load
    parser.add_argument('--save_path', type=str, default="./logits/", help="save the logits")
    parser.add_argument('--load_model', type=str, default="./savedmodels/",
                        help="load model")
    # hyper-params
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    args = parser.parse_args()

    args.load_model += args.dataset + '/'
    torch.manual_seed(0)
    np.random.seed(0)

    # Setup Dataloader
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
    if args.dataset == 'pascal':
        train_data = pascalVOCLoader('./datasets/pascal/',
                                     img_transform=img_transform, label_transform=label_transform)
        val_data = pascalVOCLoader('./datasets/pascal/', split="voc12-val",
                                   img_transform=img_transform, label_transform=label_transform)

    elif args.dataset == 'coco':
        train_data = cocoloader("./datasets/coco/",
                                img_transform=img_transform, label_transform=label_transform)
        val_data = cocoloader('./datasets/coco/', split="multi-label-val2014",
                              img_transform=img_transform, label_transform=label_transform)
    elif args.dataset == "nus-wide":
        train_data = nuswideloader("./datasets/nus-wide/",
                                   img_transform=img_transform, label_transform=label_transform)
        val_data = nuswideloader("./datasets/nus-wide/", split="val",
                                 img_transform=img_transform, label_transform=label_transform)

    # To speed up the process
    args.n_classes = val_data.n_classes
    indices = np.random.randint(len(val_data), size = 1000)
    val_data = data.Subset(val_data, indices)

    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True)
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True, pin_memory=True)

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
        model = nn.DataParallel(model).cuda()
        clsfier = nn.DataParallel(clsfier).cuda()

    # load models
    model.load_state_dict(torch.load(args.load_model + args.arch + '.pth'))
    clsfier.load_state_dict(torch.load(args.load_model + args.arch + 'clssegsimp.pth'))
    print("model loaded!")

    # freeze the batchnorm and dropout layers
    model.eval()
    clsfier.eval()

    temp = [1, 10, 100, 1000]
    noises = [0, 0.0002, 0.0004, 0.0008, 0.001, 0.0012, 0.0014,
              0.0016, 0.002, 0.0024, 0.0028, 0.003,
              0.0034, 0.0036, 0.004]
    if args.ood == "M":
        temp = [1]
        noises = [0, 0.0005, 0.001, 0.0014, 0.002, 0.005]
        # feature extraction
        temp_x = torch.rand(2, 3, 256, 256)
        temp_x = Variable(temp_x.cuda())
        temp_list = lib.model_feature_list(model, clsfier, temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        # print(feature_list)
        print('get sample mean and covariance')
        sample_mean, precision = lib.sample_estimator(model, clsfier, args.n_classes, feature_list, train_loader)

    best_T = 1
    best_noise = 0
    best_fpr = 100
    for T in temp:
        for noise in noises:
            args.T = T
            args.noise = noise
            print("T = "+str(T)+"\tnoise = "+str(noise))
            fpr = validate(args)
            if fpr < best_fpr:
                best_T = T
                best_noise = noise
                best_fpr = fpr

    f = open("./" + args.dataset + '_' + args.arch + '_' + args.ood + '_'  +
             args.method + ".txt", 'w')
    f.write("Best T%d\tBest noise%.5f\n"%(best_T, best_noise))
    f.close()

    print("Best T%d\tBest noise%.5f"%(best_T, best_noise))
