import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
import torchvision
from utils.anom_utils import ToLabel
from model.classifiersimple import *
from utils.dataloader.pascal_voc_loader import *
from utils.dataloader.nus_wide_loader import *
from utils.dataloader.coco_loader import *
import lib
import torchvision.transforms as transforms
from utils import anom_utils


def tune():
    # compute in_data score
    torch.manual_seed(0)
    np.random.seed(0)
    if args.ood == 'M':
        pack = (sample_mean, precision)
        in_scores = lib.get_Mahalanobis_score(model, clsfier, val_loader, pack,
                                              args.noise, args.n_classes, args.method)
    else:
        in_scores = lib.get_odin_scores(val_loader, model, clsfier, args.method,
                                        args.T, args.noise)

    ############################### ood ###############################
    ood_num_examples = 1000
    auroc_list = []
    aupr_list = []
    fpr_list = []
    # /////////////// Gaussion Noise ///////////////
    # print("Gaussion noise detection")
    dummy_targets = -torch.ones(ood_num_examples, args.n_classes)
    ood_data = torch.from_numpy(np.float32(np.clip(
        np.random.normal(size=(ood_num_examples, 3, 256, 256), scale=0.5), -1, 1)))
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.n_workers)
    # save_name = "\nGaussion"
    if args.ood == "M":
        out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
                                              args.noise, args.n_classes, args.method)

    else:
        out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
                                         args.T, args.noise)

    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    # f.write(save_name+'\n')
    # f.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * 0.95), 100 * fpr))
    # f.write('AUROC: \t\t\t{:.2f}\n'.format(100 * auroc))
    # f.write('AUPR:  \t\t\t{:.2f}\n'.format(100 * aupr))
    # f.write('\n')

    # /////////////// Uniform Noise ///////////////
    # print('Uniform[-1,1] Noise Detection')
    dummy_targets = -torch.ones(ood_num_examples, args.n_classes)
    ood_data = torch.from_numpy(
        np.random.uniform(size=(ood_num_examples, 3, 256, 256),
                          low=-1.0, high=1.0).astype(np.float32))
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.n_workers)

    # save_name = "\nUniform"
    if args.ood == "M":
        out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
                                               args.noise, args.n_classes, args.method)
    else:
        out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
                                         args.T, args.noise)
    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    # f.write(save_name+'\n')
    # f.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * 0.95), 100 * fpr))
    # f.write('AUROC: \t\t\t{:.2f}\n'.format(100 * auroc))
    # f.write('AUPR:  \t\t\t{:.2f}\n'.format(100 * aupr))
    # f.write('\n')

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

    # save_name = "\nArithmetic_Mean"
    # print(save_name + 'Detection')
    if args.ood == "M":
        out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
                                               args.noise, args.n_classes, args.method)
    else:
        out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
                                         args.T, args.noise)
    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    # f.write(save_name+'\n')
    # f.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * 0.95), 100 * fpr))
    # f.write('AUROC: \t\t\t{:.2f}\n'.format(100 * auroc))
    # f.write('AUPR:  \t\t\t{:.2f}\n'.format(100 * aupr))
    # f.write('\n')

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
    # save_name = "\nGeometric_Mean"
    # print(save_name + 'Detection')
    if args.ood == "M":
        out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
                                               args.noise, args.n_classes, args.method)
    else:
        out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
                                         args.T, args.noise)
    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    # f.write(save_name+'\n')
    # f.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * 0.95), 100 * fpr))
    # f.write('AUROC: \t\t\t{:.2f}\n'.format(100 * auroc))
    # f.write('AUPR:  \t\t\t{:.2f}\n'.format(100 * aupr))
    # f.write('\n')

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
    # save_name = "\nJigsaw"
    # print(save_name + 'Detection')
    if args.ood == "M":
        out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
                                               args.noise, args.n_classes, args.method)
    else:
        out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
                                         args.T, args.noise)
    auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
                                                        args.ood, args.method)
    auroc_list.append(auroc)
    aupr_list.append(aupr)
    fpr_list.append(fpr)
    # f.write(save_name+'\n')
    # f.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * 0.95), 100 * fpr))
    # f.write('AUROC: \t\t\t{:.2f}\n'.format(100 * auroc))
    # f.write('AUPR:  \t\t\t{:.2f}\n'.format(100 * aupr))
    # f.write('\n')

    # /////////////// Speckled Images ///////////////

    # speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
    # ood_loader.dataset.transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), speckle, normalize])
    # save_name = "\nSpeckled"
    # print(save_name + 'Detection')
    # if args.ood == "M":
    #     out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
    #                                            args.noise, args.n_classes)
    # else:
    #     out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
    #                                      args.T, args.noise)
    # auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
    #                                                     args.ood, args.method)
    # auroc_list.append(auroc)
    # aupr_list.append(aupr)
    # fpr_list.append(fpr)
    # # f.write(save_name+'\n')
    # # f.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * 0.95), 100 * fpr))
    # # f.write('AUROC: \t\t\t{:.2f}\n'.format(100 * auroc))
    # # f.write('AUPR:  \t\t\t{:.2f}\n'.format(100 * aupr))
    # # f.write('\n')
    #
    #
    # # /////////////// Pixelated Images ///////////////
    #
    # pixelate = lambda x: x.resize((int(256 * 0.2), int(256 * 0.2)), PILImage.BOX).resize((256, 256), PILImage.BOX)
    # ood_loader.dataset.transform = transforms.Compose([pixelate,
    #                                 transforms.ToTensor(), normalize])
    # save_name = "\nPixelated"
    # print(save_name + 'Detection')
    # if args.ood == "M":
    #     out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
    #                                            args.noise, args.n_classes)
    # else:
    #     out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
    #                                      args.T, args.noise)
    # auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
    #                                                     args.ood, args.method)
    # auroc_list.append(auroc)
    # aupr_list.append(aupr)
    # fpr_list.append(fpr)
    # # f.write(save_name+'\n')
    # # f.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * 0.95), 100 * fpr))
    # # f.write('AUROC: \t\t\t{:.2f}\n'.format(100 * auroc))
    # # f.write('AUPR:  \t\t\t{:.2f}\n'.format(100 * aupr))
    # # f.write('\n')
    #
    #
    # # /////////////// RGB Ghosted/Shifted Images ///////////////
    #
    # rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(256 - 1, -1, -1)])),
    #                                  x[2:, :, :], x[0:1, :, :]), 0)
    # ood_loader.dataset.transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),rgb_shift, normalize])
    #
    # save_name = "\nShifted"
    # print(save_name + 'Detection')
    # if args.ood == "M":
    #     out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
    #                                            args.noise, args.n_classes)
    # else:
    #     out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
    #                                      args.T, args.noise)
    # auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
    #                                                     args.ood, args.method)
    # auroc_list.append(auroc)
    # aupr_list.append(aupr)
    # fpr_list.append(fpr)
    # # f.write(save_name + '\n')
    # # f.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * 0.95), 100 * fpr))
    # # f.write('AUROC: \t\t\t{:.2f}\n'.format(100 * auroc))
    # # f.write('AUPR:  \t\t\t{:.2f}\n'.format(100 * aupr))
    # # f.write('\n')
    #
    # # /////////////// Inverted Images ///////////////
    # # not done on all channels to make image ood with higher probability
    # invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, ], 1 - x[2:, :, :],), 0)
    # ood_loader.dataset.transform = transforms.Compose([transforms.Resize((256,256)),
    #     transforms.ToTensor(),invert, normalize])
    #
    # save_name = "\nInverted"
    # print(save_name + 'Detection')
    # if args.ood == "M":
    #     out_scores = lib.get_Mahalanobis_score(model, clsfier, ood_loader, pack,
    #                                            args.noise, args.n_classes)
    # else:
    #     out_scores = lib.get_odin_scores(ood_loader, model, clsfier, args.method,
    #                                      args.T, args.noise)
    # auroc, aupr, fpr = anom_utils.get_and_print_results(in_scores, out_scores,
    #                                                     args.ood, args.method)
    # auroc_list.append(auroc)
    # aupr_list.append(aupr)
    # fpr_list.append(fpr)
    # # f.write(save_name + '\n')
    # # f.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * 0.95), 100 * fpr))
    # # f.write('AUROC: \t\t\t{:.2f}\n'.format(100 * auroc))
    # # f.write('AUPR:  \t\t\t{:.2f}\n'.format(100 * aupr))
    # # f.write('\n')

    # /////////////// Mean Results ///////////////

    # print('Mean Validation Results')
    anom_utils.print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list),
                              ood="Mean", method="validation")
    # f.write("Mean Validation Results\n")
    # f.write('FPR{:d}:\t\t\t{:.2f}\n'.format(int(100 * 0.95), 100 * np.mean(fpr_list)))
    # f.write('AUROC: \t\t\t{:.2f}\n'.format(100 * np.mean(auroc_list)))
    # f.write('AUPR:  \t\t\t{:.2f}\n'.format(100 * np.mean(aupr_list)))

    return np.mean(fpr_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    # ood measures
    parser.add_argument('--ood', type=str, default='odin',
                        help='which measure to tune odin|M')
    parser.add_argument('--method', type=str, default='max',
                        help='which method to use max|sum')
    parser.add_argument('--dataset', type=str, default='pascal',
                        help='Dataset to use pascal|coco|nus-wide')
    parser.add_argument('--arch', type=str, default='densenet',
                        help='Architecture to use densenet|resnet101')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch Size')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--n_classes', type=int, default=20)

    parser.add_argument('--load_model', type=str, default="saved_models/",
                        help="Path to load models")
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
           # test_loader = data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8, shuffle=False)
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

    # load models
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

    # load model
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
        noises = [0, 0.002, 0.0014, 0.001, 0.0005, 0.005]
        # feature extraction
        temp_x = torch.rand(2, 3, 256, 256)
        temp_x = Variable(temp_x.cuda())
        temp_list = lib.model_feature_list(model, clsfier, temp_x, args.arch)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        # print(feature_list)
        print('get sample mean and covariance')
        #
        sample_mean, precision = lib.sample_estimator(model, clsfier, args.n_classes, feature_list, train_loader)

    best_T = 1
    best_noise = 0
    best_fpr = 100
    for T in temp:
        for noise in noises:
            args.T = T
            args.noise = noise
            print("T = "+str(T)+"\tnoise = "+str(noise))
            fpr = tune()
            if fpr < best_fpr:
                best_T = T
                best_noise = noise
                best_fpr = fpr

    f = open("./" + args.dataset + '_' + args.arch + '_' + args.ood + '_' +
             args.method + ".txt", 'w')
    f.write("Best T%d\tBest noise%.5f\n" % (best_T, best_noise))
    f.close()

    print("Best T%d\tBest noise%.5f" % (best_T, best_noise))
