from os.path import join

"""CONSTANTS"""
DATASET_DIR: str = '/data/fusang/fm/celeba_raw'
IMG_DIR: str = join(DATASET_DIR, 'img_align_celeba')
PARTITION_FILE: str = join(DATASET_DIR, 'Anno/list_eval_partition.txt')
ATTRIBUTE_FILE: str = join(DATASET_DIR, 'Anno/list_attr_celeba.txt')
TRAIN_ATTRIBUTE_LIST: str = join(DATASET_DIR, 'Anno/train_attr_list.txt')
VAL_ATTRIBUTE_LIST: str = join(DATASET_DIR, 'Anno/val_attr_list.txt')
TEST_ATTRIBUTE_LIST: str = join(DATASET_DIR, 'Anno/test_attr_list.txt')
CHECKPOINT_DIR: str = 'checkpoints'
BACKUP_DIR: str = 'backups'
# TESTSET_DIR: str = '/data/fusang/fm/celeba_raw/celeba_19_31_34_no7_train_4_1'
# TESTSET_DIR: str = '/data/fusang/fm/celeba_raw/img_align_celeba'
TESTSET_DIR: str = '/data/fusang/fm/style_mix/19_31_34_no7_10K'
INFERENCE_DIR: str = 'inf'
    
"""HYPER PARAMETERS"""
# Miscs
manual_seed = 42 #1903
evaluate = False
gpu_id = '0'
disable_tqdm = True
auto_hibernate = True

# optimization
train_batch = 256 #100
dl_workers = 8
test_batch = 128 #128
epochs = 80 #60
lr = 0.01 #0.1, 0.01
lr_decay = 'step' #step, cos, linear, linear2exp, schedule
step = 30 # interval for learning rate decay in step mode
schedule = [30, 35, 40, 45, 50, 55, 56, 57, 58, 59, 60] # decrease learning rate at these epochs [150, 225]
turning_point = 100 # epoch number from linear to exponential decay mode
gamma = 0.1 #LR is multiplied by gamma on schedule 0.1
momentum = 0.9
weight_decay = 1e-4  #1e-4 
criterion = 'FocalLoss' #FocalLoss CE
optimizer = 'SGD' #SGD, Adam, AdamW
scheduler = 'ReduceLROnPlateau' #Manual ReduceLROnPlateau OneCycleLR CosineWarmupLR
patience = 5 # patience for ReduceLROnPlateau scheduler
no_bias_bn_decay = True # Turn off bias decay (default: True)
label_smoothing = 0 # 0 to turn off, 0.1 (default)
mixed_up = 0.2 # mixedup alpha value: 0 to turn off, 0.2 (default)

# Early Stopping
early_stopping = True
es_min = 30 # minimum patience
es_patience = 10 

# Checkpoints and loggers
ckp_resume = '' #path to latest checkpoint (default: none) #join(CHECKPOINT_DIR, 'checkpoint.pth.tar')
ckp_logger_fname = join(CHECKPOINT_DIR, 'log.txt')
checkpoint_fname = join(CHECKPOINT_DIR, 'checkpoint.pth.tar')
bestmodel_fname = join(CHECKPOINT_DIR, 'model_best.pth.tar')
tensorboard_dir = 'runs'
train_plotfig = join(CHECKPOINT_DIR, 'logs.eps')
train_saveplot = True
test_preds_fname = join(CHECKPOINT_DIR, 'test_preds.json')

# Architecture
arch = 'FaceAttrResNeXt' # #model architecture: FaceAttrResNet FaceAttrMobileNetV2 FaceAttrResNeXt FaceAttrMobileNetV2'
pt_layers = 50 # 18, 34, 50
