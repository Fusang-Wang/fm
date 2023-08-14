

nb_ver = 1.5
title = f'ai6126-p1-train-v{nb_ver}'
print(title)
comments = "54"
print(comments)

import os, glob
local_download_path = '/data/fusang/fm/celeba_raw/img_align_celeba'
download_dataset = True
if os.path.exists(local_download_path):
    images = glob.glob(local_download_path + '/*.jpg')
    if len(images) == 202599:
        download_dataset = False
print(f"download celeba dataset: {download_dataset}")

if download_dataset:
    print("please check dataset integrity")

import sys, os
import shutil
import time
import random
import numpy as np
import copy
from datetime import datetime
from distutils.dir_util import copy_tree #for recursive filecopying
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from celeba_dataset import CelebaDataset
import models
import losses
#import schedulers
from utils import Logger, AverageMeter, Bar, ModelTimer, savefig, adjust_learning_rate, accuracy, print_attribute_acc, create_dir_ifne, add_weight_decay, mixup_data

print(torch.__version__, torch.cuda.is_available())

# define device
device = torch.device("cuda:"+config.gpu_id if torch.cuda.is_available() else "cpu")
print(device)

ISJUPYTER = False
if 'ipykernel' in sys.modules:
    ISJUPYTER = True
    # set the backend of matplotlib to the 'inline' backend
    config.disable_tqdm = False
    
print(f"disable_tqdm: {config.disable_tqdm}")

# %% [markdown]
# ### Seeding

# %%
# set random seed for reproducibility
# self.classes = np.array([19, 31, 34, 15, 35])
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
    
seed_everything(seed=config.manual_seed)

# %% [markdown]
# ### Data Transform

# %%
# Data augmentation and normalization for training
# Just normalization for validation and testing
def load_dataloaders(print_info=True, albu_transforms = True):
    if config.evaluate:
        phases = ['test']
    else:
        phases = ['train', 'val']

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
                # transforms.CenterCrop((158, 158)), #new
                # transforms.Resize((128, 128)),
                transforms.CenterCrop((128, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomRotation(degrees=10), #new
                transforms.ToTensor(),
                normalize,
            ]),
            'val': transforms.Compose([
                # transforms.CenterCrop((158, 158)), #new
                # transforms.Resize((128, 128)),
                transforms.CenterCrop((128, 128)),
                transforms.ToTensor(),
                normalize
            ]),
            'test': transforms.Compose([
                # transforms.CenterCrop((158, 158)), #new
                # transforms.Resize((128, 128)),
                transforms.CenterCrop((128, 128)),
                transforms.ToTensor(),
                normalize
            ])
        }
    else:
        normalize_A = A.Normalize(mean=(0.485, 0.456, 0.406), 
                                  std=(0.229, 0.224, 0.225))
        data_transforms = {
            'train': A.Compose([
                #A.RandomResizedCrop(148, 148), # cuts out too much attributes, use centercrop instead
                # A.CenterCrop(height=158, width=158),
                # A.Resize(height=128, width=128),
                A.CenterCrop(height=128, width=128),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                                 rotate_limit=15, p=0.5), # AFFACT https://arxiv.org/pdf/1611.06158.pdf
                A.HorizontalFlip(p=0.5),
                #A.HueSaturationValue(hue_shift_limit=14, sat_shift_limit=14, val_shift_limit=14, p=0.5),
                #A.FancyPCA(alpha=0.1, p=0.5), #http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
                #A.RandomBrightnessContrast(p=0.5),
                #A.GaussNoise(var_limit=10.0, p=0.5), 
                #A.GaussianBlur(p=0.1), # AFFACT https://arxiv.org/pdf/1611.06158.pdf
                #A.CoarseDropout(max_holes=1, max_height=74, max_width=74, 
                #               min_height=49, min_width=49, fill_value=0, p=0.2), #https://arxiv.org/pdf/1708.04896.pdf
                normalize_A,
                ToTensorV2(),
                
            ]),
            'val': A.Compose([
                #Rescale an image so that minimum side is equal to max_size 178 (shortest edge of Celeba)
                #A.SmallestMaxSize(max_size=178), 
                # A.CenterCrop(height=158, width=158),
                # A.Resize(height=128, width=128),
                A.CenterCrop(height=128, width=128),
                normalize_A,
                ToTensorV2(),
            ]),
            'test': A.Compose([
                #A.SmallestMaxSize(max_size=178),
                # A.CenterCrop(height=158, width=158),
                # A.Resize(height=128, width=128),
                A.CenterCrop(height=128, width=128),
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
        
    if config.evaluate:
        class_names = image_datasets['test'].targets
    else:
        class_names = image_datasets['train'].targets
    # print(f"Class Labels: {len(class_names[0])}")
    # print(attribute_names, class_names[0])
    assert len(attribute_names) == len(class_names[0])
    return dataloaders, attribute_names

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name])) # and name.islower() and not name.startswith("__"))
print(f"Available Models: {model_names}")

def create_model(arch, layers, device):
    print("=> creating model '{}'".format(config.arch))
    if arch.startswith('FaceAttrResNet'):
        model = models.__dict__[arch](resnet_layers = layers)
    elif arch.startswith('FaceAttrResNeXt'):
        model = models.__dict__[arch](resnet_layers = layers)
    elif arch.startswith('FaceAttrMobileNetV2'):
        model = models.__dict__[arch]()
    model = model.to(device)
    return model

model = create_model(config.arch, config.pt_layers, device)

def get_criterion():
    criterion = nn.CrossEntropyLoss().to(device)
    if config.criterion == 'CE' and config.label_smoothing:
        criterion = losses.LabelSmoothingCrossEntropy(ls=config.label_smoothing).to(device) 
    elif config.criterion == 'FocalLoss':
        criterion = losses.FocalLossLS(alpha=0.25, gamma=3, reduction='mean', ls=config.label_smoothing).to(device) 
        
    if config.mixed_up > 0:
        criterion = losses.MixedUp(criterion).to(device) 
        
    return criterion

criterion = get_criterion()

def get_optimizer(model, opt_name=config.optimizer, no_bias_bn_decay=config.no_bias_bn_decay):
    weight_decay = config.weight_decay
    if no_bias_bn_decay: #bag of tricks paper
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()
    
    optimizer = None
    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(parameters, config.lr,
                                momentum=config.momentum,
                                weight_decay=weight_decay)
    elif opt_name == 'Adam':
        optimizer = torch.optim.Adam(parameters, config.lr,
                            weight_decay=weight_decay)
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, config.lr,
                            weight_decay=weight_decay)
    return optimizer

def get_scheduler(optimizer, steps_per_epoch, epochs):
    scheduler = None # Manual
    if config.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               factor=0.1,
                                                               patience=config.patience)
    elif config.scheduler == 'OneCycleLR': 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=epochs,
                                                        steps_per_epoch=int(steps_per_epoch), 
                                                        anneal_strategy='cos') #https://arxiv.org/pdf/1708.07120.pdf
#     elif config.scheduler == 'CosineWarmupLR':
#         scheduler = schedulers.CosineWarmupLR(optimizer, batches=int(steps_per_epoch),
#                                               epochs=epochs, base_lr=0.001, target_lr=0, warmup_epochs=5,
#                                               warmup_lr = 0.01)
    
    return scheduler    

def format_checkpoint(modelname, opt_name, bias_decay=False, ckp_resume=None):
    best_prec1 = 0

    if ckp_resume and os.path.isfile(ckp_resume): 
        print(f"=> formatting model: {ckp_resume}")
        checkpoint = torch.load(ckp_resume)
        print(checkpoint['arch'])
        try:
            total_time = checkpoint['total_time']
        except:
            total_time = 0
        
        state = {
            'epoch': checkpoint['epoch'],
            'arch': modelname,
            'state_dict': checkpoint['state_dict'],
            'best_prec1': checkpoint['best_prec1'],
            'opt_name': opt_name,
            'optimizer' : checkpoint['optimizer'],
            'lr': checkpoint['lr'],
            'total_time': total_time,
            'bias_decay': bias_decay
        }
        torch.save(state, ckp_resume)
        
    else:
        raise
        
def resume_checkpoint(device, ckp_logger_fname, ckp_resume=None):
    if not ckp_logger_fname:
        print("[W] Logger path not found.")
        raise

    start_epoch = 0
    best_prec1 = 0
    lr = config.lr
    
    if ckp_resume == '':
        ckp_resume = None
    
    if ckp_resume and os.path.isfile(ckp_resume): 
        print(f"=> resuming checkpoint: {ckp_resume}")
        checkpoint = torch.load(ckp_resume)
        
        try:
            total_time = checkpoint['total_time']
            model_timer = ModelTimer(total_time)
            print(f"=> model trained time: {model_timer}")
        except:
            print(f"=> old model")
            model_timer = ModelTimer()
        best_prec1 = checkpoint['best_prec1']
        print(f"=> model best val: {best_prec1}")
        
        start_epoch = checkpoint['epoch']
        print(f"=> model epoch: {start_epoch}")
        lr = checkpoint['lr']

        print(f"=> resuming model: {checkpoint['arch']}")
        model = create_model(checkpoint['arch'].split('_')[0], 
                             int(checkpoint['arch'].split('_')[1]), 
                             device)
        model.load_state_dict(checkpoint['state_dict'])
        
        print(f"=> resuming optimizer: {checkpoint['opt_name']}")
        bias_decay = True
        if checkpoint['bias_decay']:
            bias_decay = checkpoint['bias_decay']
            
        optimizer = get_optimizer(model, checkpoint['opt_name'], bias_decay)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(ckp_logger_fname, title=model.name, resume=True)
        
    else:
        print(f"=> restarting training: {ckp_resume}")
        model_timer = ModelTimer()
        model = create_model(config.arch, config.pt_layers, device)
        optimizer = get_optimizer(model)
        logger = Logger(ckp_logger_fname, title=model.name)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
              
    return best_prec1, model_timer, lr, start_epoch, logger, model, optimizer

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
    start_epoch = checkpoint['epoch']
    print(f"=> model epoch: {start_epoch}")

    print(f"=> resuming model: {checkpoint['arch']}")
    model = create_model(checkpoint['arch'].split('_')[0], 
                         int(checkpoint['arch'].split('_')[1]), 
                         device)
    model.load_state_dict(checkpoint['state_dict'])
              
    return best_prec1, model

def train(train_loader, model, criterion, optimizer):
    bar = Bar('Processing', max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(len(classes))]
    top1 = [AverageMeter() for _ in range(len(classes))]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (X, y) in enumerate(tqdm(train_loader, disable=config.disable_tqdm)):
        # measure data loading time
        data_time.update(time.time() - end)

        # Overlapping transfer if pinned memory
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if config.mixed_up > 0:
            X, y, lam = mixup_data(X, y, alpha=config.mixed_up)
            criterion.set_lambda(lam)
    
        # compute output
        output = model(X)
        # measure accuracy and record loss
        loss = []
        prec1 = []
        for j in range(len(output)): 
            if config.mixed_up > 0:
                labels = y[:, :, j]
                actual_labels = y[0, :, j] * lam + y[1, :, j] * (1-lam)
            else:
                labels = y[:, j]
                actual_labels = y[:, j]
            crit = criterion(output[j], labels)
            loss.append(crit)
            prec1.append(accuracy(output[j], actual_labels, topk=(1,), mixedup=config.mixed_up))
            losses[j].update(loss[j].detach().item(), X.size(0))
            top1[j].update(prec1[j][0].item(), X.size(0))
            
        losses_avg = [losses[k].avg for k in range(len(losses))]
        top1_avg = [top1[k].avg for k in range(len(top1))]
        loss_avg = sum(losses_avg) / len(losses_avg)
        prec1_avg = sum(top1_avg) / len(top1_avg)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss_sum = sum(loss)
        loss_sum.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        print_line = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                        batch=i + 1,
                        size=len(train_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=loss_avg,
                        top1=prec1_avg,
                        )
        if not config.disable_tqdm and (i+1)% 100 == 0:
            print(print_line)
        bar.suffix  = print_line
        bar.next()
    bar.finish()
    return (loss_avg, prec1_avg)

def validate(val_loader, model, criterion):
    bar = Bar('Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(len(classes))]
    top1 = [AverageMeter() for _ in range((len(classes)))]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (X, y) in enumerate(tqdm(val_loader, disable=config.disable_tqdm)):
            # measure data loading time
            data_time.update(time.time() - end)

            # Overlapping transfer if pinned memory
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # compute output
            output = model(X)
            # measure accuracy and record loss
            loss = []
            prec1 = []
            for j in range(len(output)):
                if config.mixed_up > 0:
                    loss.append(criterion(output[j], y[:, j], mixed=False))
                else:
                    loss.append(criterion(output[j], y[:, j]))
                prec1.append(accuracy(output[j], y[:, j], topk=(1,)))
                
                losses[j].update(loss[j].detach().item(), X.size(0))
                top1[j].update(prec1[j][0].item(), X.size(0))
            losses_avg = [losses[k].avg for k in range(len(losses))]
            top1_avg = [top1[k].avg for k in range(len(top1))]
            loss_avg = sum(losses_avg) / len(losses_avg)
            prec1_avg = sum(top1_avg) / len(top1_avg)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # plot progress
            print_line = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=loss_avg,
                            top1=prec1_avg,
                            )

            bar.suffix  = print_line
            bar.next()  

    if not config.disable_tqdm:
        print(print_line)        
    bar.finish()
    return (loss_avg, prec1_avg, top1)

# %% [markdown]
# ## Main Function

# %%
def trainer(dataloaders, model, criterion, optimizer, logger, start_epoch, best_prec1, run_name, model_timer):
    # visualization
    writer = SummaryWriter(os.path.join(config.tensorboard_dir, run_name))
    
    scheduler = get_scheduler(optimizer, len(dataloaders['train']), config.epochs-start_epoch)
    
    stagnant_val_loss_ctr = 0
    min_val_loss = 1.
    
    for epoch in range(start_epoch, config.epochs):
        model_timer.start_epoch_timer()
        if not scheduler:
            lr = adjust_learning_rate(optimizer, config.lr_decay, epoch, gamma=config.gamma, step=config.step,
                                     total_epochs=config.epochs, turning_point=config.turning_point,
                                     schedule=config.schedule)
        else:
            lr = optimizer.param_groups[0]['lr']

        print('\nEpoch: [%d | %d] LR: %.16f' % (epoch + 1, config.epochs, lr))

        # train for one epoch
        train_loss, train_acc = train(dataloaders['train'], model, criterion, optimizer)

        # evaluate on validation set
        val_loss, prec1, _ = validate(dataloaders['val'], model, criterion)
        
        if scheduler:
            scheduler.step(None if config.scheduler != 'ReduceLROnPlateau' else val_loss)
            
        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        model_timer.stop_epoch_timer()
        model.save_ckp({
            'epoch': epoch + 1,
            'arch': model.name,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'opt_name': config.optimizer,
            'optimizer' : optimizer.state_dict(),
            'lr': lr,
            'total_time': model_timer.total_time,
            'bias_decay': config.no_bias_bn_decay,
        }, is_best, config.checkpoint_fname,config.bestmodel_fname)
        
        if config.early_stopping:
            if is_best:
                stagnant_val_loss_ctr = 0
                min_val_loss = val_loss
            elif val_loss >= min_val_loss:
                stagnant_val_loss_ctr += 1
                if (epoch+1) > config.es_min and stagnant_val_loss_ctr >= config.es_patience: 
                    break
            else:
                stagnant_val_loss_ctr = 0
                min_val_loss = val_loss

    print("training completed")
    logger.close()
    writer.close()
    try:
        print("normal plot")
        logger.plot()
        save_path = None
        if config.train_saveplot:
            save_path = os.path.join(config.CHECKPOINT_DIR, "losses.jpg")
        print("special plot")
        logger.plot_special(save_path)
        savefig(config.train_plotfig)
    except:
        print("error plotting")
    
    print('Best accuracy:')
    print(best_prec1)
    return model_timer

# %%
def get_run_name_time(model, criterion, optimizer, comments, start_epoch=0):
    try:
        if criterion.name:
            p_criterion = criterion.name
    except:
        p_criterion = 'CE'

    p_optimizer = f'{str(optimizer).split("(")[0].strip()}'
    p_scheduler = f'lr{config.lr}_wd{config.weight_decay}'
    if config.scheduler == 'Manual':
        p_scheduler += f'_{config.lr_decay}'
        if config.lr_decay == 'step':
            p_scheduler += f'_g{config.gamma}_sp{config.step}'
        elif config.lr_decay == 'linear2exp':
            p_scheduler += f'_g{config.gamma}_tp{config.turning_point}'
        elif config.lr_decay == 'schedule':
            p_scheduler += f'_g{config.gamma}_sch{config.schedule}'
    else: 
        p_scheduler += f'_{config.scheduler}'
    
    run_name = f'{model.name}_{config.manual_seed}_s{start_epoch}e{config.epochs}_' \
                + f'tb{config.train_batch}_vb{config.test_batch}_' \
                + f'{p_criterion}_{p_optimizer}_' \
                + f'{comments}_' \
                + f'{p_scheduler}'
    
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(run_name, run_time)
    return run_name, run_time

# %% [markdown]
# ## Training Loop

# %%
# config.epoch = 1
#model = create_model(device)
dataloaders, attribute_names = load_dataloaders()
criterion = get_criterion()
#optimizer = get_optimizer(model)

print(f"=> Training model: {not config.evaluate}")
if config.evaluate:
    best_prec1, model = load_inference_model(device, config.bestmodel_fname) # checkpoint_fname bestmodel_fname
    test_loss, prec1, top1 = validate(dataloaders['test'], model, criterion)
    print(f"=> Best test accuracy: {prec1}, Model val acc: {best_prec1}")
    attr_acc = print_attribute_acc(top1, attribute_names)
    if config.test_preds_fname:
        json.dump(attr_acc, open(config.test_preds_fname,'w'))
else:
    best_prec1, model_timer, lr, start_epoch, logger, model, optimizer = resume_checkpoint(device, config.ckp_logger_fname, config.ckp_resume)
    run_name, run_time = get_run_name_time(model, criterion, optimizer, comments, start_epoch)
    mtimer = trainer(dataloaders, model, criterion, optimizer, logger, start_epoch, best_prec1, run_name, model_timer)
    print(f"=> Model trained time: {mtimer}")


# %% [markdown]
# ## Testing Loop

# %%
if not config.evaluate:
    config.evaluate = True
    #model = create_model(device)
    dataloaders, attribute_names = load_dataloaders()
    criterion = get_criterion()
    #optimizer = get_optimizer(model)
    
    best_prec1, model = load_inference_model(device, config.bestmodel_fname) # checkpoint_fname bestmodel_fname
    #best_prec1, mtimer, _, _, logger, = resume_checkpoint(model, optimizer, config.ckp_logger_fname, config.checkpoint_fname)
    test_loss, prec1, top1 = validate(dataloaders['test'], model, criterion)
    print(f"=> Best test accuracy: {prec1}, Model val acc: {best_prec1}")
    attr_acc = print_attribute_acc(top1, attribute_names)
    if config.test_preds_fname:
        json.dump(attr_acc, open(config.test_preds_fname,'w'))
#     best_prec1, mtimer, _, _, _, = resume_checkpoint(model, optimizer, config.ckp_logger_fname, config.bestmodel_fname)# config.bestmodel_fname  config.checkpoint_fname
#     #print(model)
#     test_loss, prec1, top1 = validate(dataloaders['test'], model, criterion)
#     print(f"=> Best test accuracy: {prec1}, Model val acc: {best_prec1}")
#     print_attribute_acc(top1, attribute_names)