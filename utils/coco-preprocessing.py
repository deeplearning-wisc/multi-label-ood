# coding: utf-8

from pycocotools.coco import COCO
import argparse
import numpy as np
#import skimage.io as io
import pylab
import os, os.path
import pickle
from tqdm import tqdm

#pylab.rcParams['figure.figsize'] = (10.0, 8.0)

parser = argparse.ArgumentParser(description="Preprocess COCO Labels.")

#dataDir='/share/data/vision-greg/coco'
#which dataset to extract options are [all, train, val, test]
#dataset = "all"
parser.add_argument("--dir", type=str, default="/nobackup-slow/dataset/coco/",
                    help="where is the coco dataset located.")
parser.add_argument("--save_dir", type=str, default="./datasets/coco/",
                    help="where to save the coco labels.")
parser.add_argument("--dataset", type=str, default="all",
                    choices=["all", "train", "val", "test"],
                    help="which coco partition to create the multilabel set" 
                    "for the options [all, train, val, test] default is all")
args = parser.parse_args()


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def wrrite(fname, d):
    fout = open(fname, 'w')
    for i in range(len(d)):
        fout.write(d[i] +'\n')
    fout.close()

def load(fname):
    data = []
    labels = []
    for line in open(fname).readlines():
        l = line.strip().split(' ')
        data.append(l[0])
        labels.append(int(l[1]))
    return data,np.array(labels,dtype=np.int32)


def load_labels(img_names, root_dir, dataset, coco, idmapper):
    labels = {}
    for i in tqdm(range(len(img_names))):
        #print(i, dataset)
        #print(img_names[i], img_names[i][18:-4])
        # Hack to extract the image id from the image name
        if dataset == "val":
            imgIds=int(img_names[i][18:-4])
        else:
            imgIds=int(img_names[i][19:-4])  
        annIds = coco.getAnnIds(imgIds=imgIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        c = []
        for annot in anns:
            c.append(idmapper[annot['category_id']])
        if not c:
            c = np.array(-1)
        labels[root_dir + '/' + img_names[i]] = np.unique(c)

    return labels


def load_image_names(root_dir):
    DIR = root_dir
    #print(DIR)
    img_names = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
    return img_names


def load_annotations(dataDir, dataType):
    annFile='%sannotations/instances_%s.json'%(dataDir, dataType)
    
    # initialize COCO api for instance annotations
    coco=COCO(annFile)
    
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    
    
    nms=[cat['id'] for cat in cats]
    idmapper = {}
    for i in range(len(nms)):
        idmapper[nms[i]] = i

    return coco, idmapper


root_dir = args.dir + "train2014"
train_img_names = load_image_names(root_dir)
root_dir = args.dir + "val2014"
val_img_names = load_image_names(root_dir)

if args.dataset == "test" or args.dataset == "all":
    root_dir = args.dir + "test2014"
    test_img_names = load_image_names(root_dir)

    d = {}
    for i in range(len(test_img_names)):
        d[i] = root_dir + '/' + test_img_names[i]

    LIST = args.save_dir + 'test2014imgs.txt'
    wrrite(LIST,d)


if args.dataset == "all":
    root_dir = args.dir + "train2014"

    coco, idmapper = load_annotations(args.dir, "train2014")
    labels = load_labels(train_img_names, root_dir, "train", coco, idmapper)
    save_obj(labels, args.save_dir + "/multi-label-train2014") 
    LIST = args.save_dir + "train2014imgs.txt"
    wrrite(LIST, train_img_names)

    root_dir = args.dir + "val2014"

    coco, idmapper = load_annotations(args.dir, "val2014")
    labels = load_labels(val_img_names, root_dir, "val", coco, idmapper)
    save_obj(labels, args.save_dir + "/multi-label-val2014")
    LIST = args.save_dir + "/val2014imgs.txt"
    wrrite(LIST, val_img_names)

elif args.dataset == 'val':

    root_dir = args.dir + "val2014"

    coco, idmapper = load_annotations(root_dir)

    labels = load_labels(val_img_names, root_dir, "val", coco, idmapper)
    save_obj(labels, args.save_dir + "/multi-label-val2014")
    LIST = args.save_dir + "/val2014imgs.txt"
    wrrite(LIST, val_img_names)


elif args.dataset == 'train':
    root_dir = args.dir + "/train2014"

    coco, idmapper = load_annotations(root_dir)

    labels = load_labels(train_img_names, root_dir, "train", coco, idmapper)
    save_obj(labels, args.save_dir + "/multi-label-train2014")
    LIST = args.save_dir + "/train2014imgs.txt"
    wrrite(LIST, train_img_names)



# For image segmentaion 
# converting polygon and RLE to binary mask

#labels = {}
#for i in range(len(imgsname)):
#    print(i)
#    if val == True:
#        imgIds=int(imgsname[i][19:25])
#    else:
#        imgIds=int(imgsname[i][21:27])  
#    annIds = coco.getAnnIds(imgIds=imgIds, iscrowd=None)
#    anns = coco.loadAnns(annIds)
#    for annot in anns:
#        cmask_partial = coco.annToMask(annot)
#
