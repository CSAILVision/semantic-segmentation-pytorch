# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from dataset import TrainDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, parse_devices, setup_logger
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata
from config import cfg


# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        batch_data = next(iterator)
        data_time.update(time.time() - tic)

        segmentation_module.zero_grad()

        # forward pass
        loss, acc = segmentation_module(batch_data)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)


def checkpoint(nets, history, cfg, epoch_num):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    # dict_encoder_save = {k: v for k, v in dict_encoder.items() if not (k.endswith('_tmp_running_mean') or k.endswith('tmp_running_var'))}
    # dict_decoder_save = {k: v for k, v in dict_decoder.items() if not (k.endswith('_tmp_running_mean') or k.endswith('tmp_running_var'))}

    torch.save(history,
               '{}/history_{}'.format(cfg.TRAIN.ckpt, suffix_latest))
    torch.save(dict_encoder,
               '{}/encoder_{}'.format(cfg.TRAIN.ckpt, suffix_latest))
    torch.save(dict_decoder,
               '{}/decoder_{}'.format(cfg.TRAIN.ckpt, suffix_latest))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)

    crit = nn.NLLLoss(ignore_index=-1)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.TRAIN,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torchdata.DataLoader(
        dataset_train,
        batch_size=len(cfg.GPUS),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if len(cfg.GPUS) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=cfg.GPUS)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch + 1):
        train(segmentation_module, iterator_train, optimizers, history, epoch, cfg)

        # checkpointing
        checkpoint(nets, history, cfg, epoch)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Parse gpu ids
    all_gpus = parse_devices(args.gpus)
    all_gpus = [x.replace('gpu', '') for x in all_gpus]
    cfg.GPUS = [int(x) for x in all_gpus]
    num_gpus = len(cfg.GPUS)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # Model ID
    cfg.MODEL.id += '-' + cfg.MODEL.arch_encoder
    cfg.MODEL.id += '-' + cfg.MODEL.arch_decoder
    cfg.MODEL.id += '-ngpus' + str(num_gpus)
    cfg.MODEL.id += '-batchSize' + str(cfg.TRAIN.batch_size)
    cfg.MODEL.id += '-imgMaxSize' + str(cfg.TRAIN.imgMaxSize)
    cfg.MODEL.id += '-paddingConst' + str(cfg.TRAIN.padding_constant)
    cfg.MODEL.id += '-segmDownsampleRate' + str(cfg.TRAIN.segm_downsampling_rate)
    cfg.MODEL.id += '-LR_encoder' + str(cfg.TRAIN.lr_encoder)
    cfg.MODEL.id += '-LR_decoder' + str(cfg.TRAIN.lr_decoder)
    cfg.MODEL.id += '-epoch' + str(cfg.TRAIN.num_epoch)
    if cfg.TRAIN.fix_bn:
        cfg.MODEL.id += '-fixBN'
    print('Model ID: {}'.format(cfg.MODEL.id))

    cfg.TRAIN.ckpt = os.path.join(cfg.TRAIN.ckpt, cfg.MODEL.id)
    if not os.path.isdir(cfg.TRAIN.ckpt):
        os.makedirs(cfg.TRAIN.ckpt)

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg)
