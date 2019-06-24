from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = "./data/training.odgt"
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.num_class = 150

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# a name for identifying the model
_C.MODEL.id = "baseline"
# architecture of net_encoder
_C.MODEL.arch_encoder = "resnet50dilated"
# architecture of net_decoder
_C.MODEL.arch_decoder = "ppm_deepsup"
# weights to finetune net_encoder
_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
_C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2
# epochs to train for
_C.TRAIN.num_epoch = 20
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 1
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16

# input image size of short edge (int or tuple)
_C.TRAIN.imgSize = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.TRAIN.imgMaxSize = 1000
# maxmimum downsampling rate of the network
_C.TRAIN.padding_constant = 8
# downsampling rate of the segmentation label
_C.TRAIN.segm_downsampling_rate = 8
# if horizontally flip images when training
_C.TRAIN.random_flip = True

# folder to output checkpoints
_C.TRAIN.ckpt = "./ckpt"
# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 304

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
