nb_ver = 0.3
title = f'ai6126-p1-inference-v{nb_ver}'
print(title)

import sys, os
import shutil
import time
import random
import numpy as np
import copy
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import json
from pprint import pprint

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from tqdm import tqdm
import config
from celeba_dataset import CelebaDataset, CelebaTestset
import models
from utils import Logger, ModelTimer, AverageMeter, accuracy, print_attribute_acc

import seaborn as sns
sns.set()

print(torch.__version__, torch.cuda.is_available())

# define device
device = torch.device("cuda:"+config.gpu_id if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # force cpu
print(device)
print(f"disable_tqdm: {config.disable_tqdm}")
classes = np.array([19, 31, 34])
def seed_everything(seed=None):
    if seed is None:
        seed = random.randint(1, 10000) # create random seed
        print(f'random seed used: {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if 'torch' in sys.modules:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
seed_everything(seed=config.manual_seed)#config.manual_seed

IMAGE_H = 128 #158 218 148 198
IMAGE_W = 128 #178 148 158

# Data augmentation and normalization for training
# Just normalization for validation and testing
def load_dataloaders(print_info=True, albu_transforms = True, img_h=158, img_w=158):
    phases = ['val', 'test'] #'train'

    attribute_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
                       'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 
                       'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                       'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 
                       'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
                       'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    attribute_names = np.array(attribute_names)[classes]

    attributes_list = {
        'train': config.TRAIN_ATTRIBUTE_LIST,
        'val': config.VAL_ATTRIBUTE_LIST,
        'test': config.TEST_ATTRIBUTE_LIST
    }

    batch_sizes = {
        'train': config.train_batch,
        'val': config.test_batch,
        'test': config.test_batch
    }

    if not albu_transforms:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop((img_h, img_w)), #new
                # transforms.Resize((128,128)),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomRotation(degrees=10), #new
                transforms.ToTensor(),
                normalize,
                #transforms.RandomErasing()
            ]),
            'val': transforms.Compose([
                #transforms.Resize(178), #new
                transforms.CenterCrop((img_h, img_w)),
                # transforms.Resize((128,128)),
                transforms.ToTensor(),
                normalize
            ]),
            'test': transforms.Compose([
                #transforms.Resize(178), #new
                transforms.CenterCrop((img_h, img_w)),
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                normalize
            ])
        }
    else:
        normalize_A = A.Normalize(mean=(0.485, 0.456, 0.406), 
                                  std=(0.229, 0.224, 0.225))
        data_transforms = {
            'train': A.Compose([
                A.CenterCrop(height=img_h, width=img_w),
                # A.Resize(height=128, width=128),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                                 rotate_limit=15, p=0.5), # AFFACT https://arxiv.org/pdf/1611.06158.pdf
                A.HorizontalFlip(p=0.5),
                #A.HueSaturationValue(hue_shift_limit=14, sat_shift_limit=14, val_shift_limit=14, p=0.5),
                #A.FancyPCA(alpha=0.1, p=0.5), #http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(var_limit=10.0, p=0.5), 
                #A.GaussianBlur(p=0.1), # AFFACT https://arxiv.org/pdf/1611.06158.pdf
                #A.CoarseDropout(max_holes=1, max_height=74, max_width=74, 
                #               min_height=49, min_width=49, fill_value=0, p=0.2), #https://arxiv.org/pdf/1708.04896.pdf
                normalize_A,
                ToTensorV2(),
                
            ]),
            'val': A.Compose([
                #Rescale an image so that minimum side is equal to max_size 178 (shortest edge of Celeba)
                #A.SmallestMaxSize(max_size=178), 
                A.CenterCrop(height=img_h, width=img_w),
                # A.Resize(height=128, width=128),
                normalize_A,
                ToTensorV2(),
            ]),
            'test': A.Compose([
                #A.SmallestMaxSize(max_size=178),
                A.CenterCrop(height=img_h, width=img_w),
                # A.Resize(height=128, width=128),
                normalize_A,
                ToTensorV2(),
            ])
        }

    image_datasets = {x: CelebaDataset(config.IMG_DIR, attributes_list[x], 
                                       data_transforms[x], albu=albu_transforms) 
                      for x in phases}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                  batch_size=batch_sizes[x],
                                                  pin_memory=True, shuffle=(x == 'train'), 
                                                  num_workers=config.dl_workers) 
                   for x in phases}
    if print_info:
        dataset_sizes = {x: len(image_datasets[x]) for x in phases}
        print(f"Dataset sizes: {dataset_sizes}")
        
    class_names = image_datasets['test'].targets
    
    print(f"Class Labels: {len(class_names[0])}")
    assert len(attribute_names) == len(class_names[0])
    return dataloaders, attribute_names

dataloaders, attribute_names = load_dataloaders(albu_transforms = True, img_h=IMAGE_H, img_w=IMAGE_W)

def load_testset(print_info=True, albu_transforms = False, img_h=158, img_w=158):    
    attribute_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
                   'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 
                   'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                   'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 
                   'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                   'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
                   'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    
    attribute_names = np.array(attribute_names)[classes]

    if not albu_transforms:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        
        test_transforms = transforms.Compose([
            transforms.CenterCrop((img_h, img_w)),
            transforms.Resize((128, 128)), 
            transforms.ToTensor(),
            normalize
        ])
        
    if albu_transforms:
        normalize_A = A.Normalize(mean=(0.485, 0.456, 0.406), 
                              std=(0.229, 0.224, 0.225))
        
        test_transforms = A.Compose([
            #A.SmallestMaxSize(max_size=178),
            A.CenterCrop(height=img_h, width=img_w),
            # A.Resize(height=128, width=128),
            normalize_A,
            ToTensorV2(),
        ]) 
        
    test_dataset = CelebaTestset(config.TESTSET_DIR, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch, 
                                             pin_memory=True, shuffle=False, num_workers=config.dl_workers)
    if print_info:
        print(f"Testset size: {len(test_dataset)}")
        print(f"Number of Celebs: {len(test_dataset.celeba_ctr.keys())}")
        
    return test_dataset, test_loader, attribute_names

test_dataset, test_loader, attribute_names = load_testset(albu_transforms = True, img_h=128, img_w=128)
# test_dataset, test_loader, attribute_names = load_testset(albu_transforms = True, img_h=158, img_w=158)
if True:
    real_batch = next(iter(test_loader))
    plt.figure(figsize=(12,12))
    plt.axis("off")
    plt.title("Private Testset Images")
    img = np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0))
    plt.imshow(img)
    plt.savefig("image0.png")

def create_model(arch, layers, device):
    print("=> creating model '{}'".format(arch))
    if arch.startswith('FaceAttrResNet'):
        model = models.__dict__[arch](resnet_layers = layers)
    elif arch.startswith('FaceAttrResNeXt'):
        model = models.__dict__[arch](resnet_layers = layers)
    elif arch.startswith('FaceAttrMobileNetV2'):
        model = models.__dict__[arch]()
    model = model.to(device)
    return model

def format_checkpoint(modelname, opt_name, bias_decay=False, ckp_resume=None):
    best_prec1 = 0

    if ckp_resume and os.path.isfile(ckp_resume): 
        print(f"=> resuming model: {ckp_resume}")
        checkpoint = torch.load(ckp_resume)
        print(checkpoint['arch'])
        try:
            total_time = checkpoint['total_time']
        except:
            total_time = 0
        try:
            lr = checkpoint['lr']
        except:
            lr = 0.1
        is_best=False
        state = {
            'epoch': checkpoint['epoch'],
            'arch': modelname,
            'state_dict': checkpoint['state_dict'],
            'best_prec1': checkpoint['best_prec1'],
            'opt_name': opt_name,
            'optimizer' : checkpoint['optimizer'],
            'lr': lr,
            'total_time': total_time,
            'bias_decay': bias_decay
        }
        torch.save(state, ckp_resume)
        
    else:
        raise

def load_inference_model(device, ckp_resume):
    if not (ckp_resume and os.path.isfile(ckp_resume)):
        print("[W] Checkpoint not found for inference.")
        raise 
    
    print(f"=> loading checkpoint: {ckp_resume}")
    checkpoint = torch.load(ckp_resume)
    try:
        total_time = checkpoint['total_time']
        model_timer = ModelTimer(total_time)
        print(f"=> model trained time: {model_timer}")
    except:
        print(f"=> old model")
    best_prec1 = checkpoint['best_prec1']
    print(f"=> model best val: {best_prec1}")

    print(f"=> resuming model: {checkpoint['arch']}")
    model = create_model(checkpoint['arch'].split('_')[0], 
                         int(checkpoint['arch'].split('_')[1]), 
                         device)
    model.load_state_dict(checkpoint['state_dict'])
              
    return best_prec1, model

# config.INFERENCE_DIR = '/data/fusang/fm/AI6126-Advanced_Computer_Vision/project_1/src/inf'
lfile = 'model_best.pth.tar'
inf_models = {}
ctr = 0
for dirt in os.listdir(config.INFERENCE_DIR):
    dirpath = os.path.join(config.INFERENCE_DIR, dirt)
    for run in os.listdir(dirpath):
        runpath = os.path.join(dirpath, run)
        if os.path.isdir(runpath): 
            for filename in os.listdir(runpath):
                if filename == lfile:
                    best_prec1, model = load_inference_model(device, os.path.join(runpath,lfile))
                    del model
                    inf_models[ctr] = (os.path.join(runpath,lfile), dirt, run, best_prec1)
                    ctr += 1
                
print(f'==> {len(inf_models)} inference model(s) found.')

# %%
keydict = {mid: (name, acc) for mid, (_, name, _, acc) in inf_models.items()}
pprint(f'{keydict}')

SAVE_FILES = True
selected_model = int(input("Enter model index: "))
p_run = inf_models[selected_model][2]
p_model_name = inf_models[selected_model][1]
run_dir = os.path.join(config.INFERENCE_DIR, p_model_name)
p_model_acc, p_model = load_inference_model(device, inf_models[selected_model][0]) 
#print(f"=> best model val: {p_model_acc}")

def validate(val_loader, model):
    top1 = [AverageMeter() for _ in range(len(classes))]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (X, y) in enumerate(tqdm(val_loader)):
            # Overlapping transfer if pinned memory
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # compute output
            output = model(X)
            # measure accuracy
            prec1 = []
            for j in range(len(output)):
                prec1.append(accuracy(output[j], y[:, j], topk=(1,)))
                top1[j].update(prec1[j][0].item(), X.size(0))

            top1_avg = [top1[k].avg for k in range(len(top1))]
            prec1_avg = sum(top1_avg) / len(top1_avg)
        
    return (prec1_avg, top1)

val_prec1, val_top1 = validate(dataloaders['val'], p_model)
print(f"=> Best val accuracy: {val_prec1}")
v_attr_acc = print_attribute_acc(val_top1, attribute_names)

test_prec1, test_top1 = validate(dataloaders['test'], p_model)
print(f"=> Best test accuracy: {test_prec1}")
test_attr_acc = print_attribute_acc(test_top1, attribute_names)

if SAVE_FILES:
    json_save_dir = os.path.join(run_dir, p_run)
    vpfile = os.path.join(json_save_dir, "val_preds.json")
    json.dump(v_attr_acc, open(vpfile,'w'))
    tpfile = os.path.join(json_save_dir, "test_preds.json")
    json.dump(test_attr_acc, open(tpfile,'w'))

del dataloaders

maxk = 1
preds = pd.DataFrame(index=test_dataset.imagenames, columns=attribute_names)
preds.index.name = "Images"
p_model.eval()

for X, names in tqdm(test_loader, disable=False):
    inputs = X.to(device, non_blocking=True)

    top_k_preds = []
    with torch.no_grad():
        outputs = p_model(inputs) # 40, BS
    
        for attr_scores in outputs:
            _, attr_preds = attr_scores.topk(maxk, 1, True, True)
            top_k_preds.append(attr_preds.t())
            
    all_preds = torch.cat(top_k_preds, dim=0) 

    all_preds = all_preds.permute(1,0).cpu()
    for j in range(len(names)):
        preds.loc[names[j], :] = all_preds[j]
num_of_classes = 3
preds_label = preds.to_numpy()
preds_label = preds_label[:, :3]
index_transformer = np.transpose([2**i for i in range(num_of_classes)])
preds_index = np.matmul(preds_label, index_transformer)

for i in range(2**num_of_classes):
    num_ = sum(preds_index == i)
    print(f"classes {i}: {num_}")

# %%
if SAVE_FILES:
    pfile = os.path.join(run_dir, "predictions.csv")
    ptxtfile = os.path.join(run_dir, "predictions.txt")
    preds.to_csv(ptxtfile, sep=' ', header=False)
    preds.to_csv(pfile, index=True)

stat_df = pd.DataFrame(index = attribute_names)
stat_df.loc[:,'Testset'] = (preds.iloc[:,:] == 1).mean(axis=0)*100
stat_df = stat_df.sort_values('Testset', ascending=False)
fig, ax = plt.subplots()
stat_df.plot(title='CelebA Private Testset Prediction Frequency Distribution', 
             kind='bar', figsize=(20, 5), ax=ax, color='green')
for p in ax.patches:
    value = round(p.get_height(),2)
    ax.annotate(str(value), xy=(p.get_x(), p.get_height()))
plt.savefig('private_test.png',dpi=160, bbox_inches='tight')

# %%
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

def get_celeb_prediction(preds, name, first_img=True):
    celeb_preds = preds[preds.index.str.contains(name)]
    celeb_first = celeb_preds.index[0]
    celeb_stat = pd.DataFrame(index = attribute_names)
    celeb_stat.loc[:,name] = (celeb_preds.iloc[:,:] == 1).mean(axis=0)*100
    mycolor = 'skyblue' if celeb_stat.loc['Male',name] >= 50 else 'magenta'
    celeb_stat = celeb_stat.sort_values(name, ascending=False)
    ncols = 3 if first_img else 2
    ax = plt.subplot2grid((1, ncols), (0, 0), colspan=2)
    celeb_stat.plot(title=name+' Prediction Frequency Distribution', 
                 kind='bar', figsize=(20, 5), color=mycolor, ax=ax)
    for p in ax.patches:
        value = round(p.get_height(),2)
        ax.annotate(str(value), xy=(p.get_x(), p.get_height()))
    if first_img:
        ax2 = plt.subplot2grid((1, ncols), (0, 2), colspan=1)
        index = test_dataset.imagenames.index(celeb_first)    
        s_img = inv_normalize(test_dataset[index][0]).permute(1, 2, 0)
        ax2.imshow(s_img)
        ax2.set_axis_off()
        plt.title(celeb_first)
        plt.tight_layout()
    plt.show()

# get_celeb_prediction(preds, name = 'George_W_Bush', first_img=True) # Male, Gray_Hair, Blonde, Necktie

def plot_prediction_with_image(preds, index=None, off_neg=True):
    if index == None:
        index = np.random.randint(0, len(preds))
        print(f"=> random index: {index}")
    
    if type(index) == int:
        p_attrs = preds.iloc[index,:]
        p_img = preds.index[index]
    else:
        p_attrs = preds.loc[index, :]
        p_img = index
        index = test_dataset.imagenames.index(index)   

    p_attrs = p_attrs.sort_values(0, ascending=True)
    fig, (ax, ax2) = plt.subplots(ncols=2)
    my_color=np.where(p_attrs>=0, 'green', 'orange')
    if off_neg:
        p_attrs[p_attrs == 1].plot(kind='bar',ax=ax, figsize=(8, 5), color=my_color)
    else:
        p_attrs.plot(kind='barh',ax=ax, figsize=(12, 8), color=my_color)
    
    s_img = inv_normalize(test_dataset[index][0]).permute(1, 2, 0)
    ax2.imshow(s_img)
    ax2.set_axis_off()
    plt.title(p_img)
    plt.show()

# plot_prediction_with_image(preds, index=201) # Young, Male, Mouth open, Nobeard, Black hair




