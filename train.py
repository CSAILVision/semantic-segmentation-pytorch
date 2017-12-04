# System libs
import os
import time
# import math
import random
import argparse
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.io import loadmat
from scipy.misc import imresize, imsave
# Our libs
from dataset import Dataset
from models import ModelBuilder
from utils import AverageMeter, colorEncode, accuracy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def forward_with_loss(nets, batch_data, args, is_train=True):
    (net_encoder, net_decoder, crit) = nets
    (imgs, segs, infos) = batch_data

    # feed input data
    input_img = Variable(imgs, volatile=not is_train)
    label_seg = Variable(segs, volatile=not is_train)
    input_img = input_img.cuda()
    label_seg = label_seg.cuda()

    # forward
    pred = net_decoder(net_encoder(input_img))
    err = crit(pred, label_seg)
    return pred, err


def visualize(batch_data, pred, args):
    colors = loadmat('data/color150.mat')['colors']
    (imgs, segs, infos) = batch_data
    for j in range(len(infos)):
        # get/recover image
        # img = imread(os.path.join(args.root_img, infos[j]))
        img = imgs[j].clone()
        for t, m, s in zip(img,
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        img = imresize(img, (args.imgSize, args.imgSize),
                       interp='bilinear')

        # segmentation
        lab = segs[j].numpy()
        lab_color = colorEncode(lab, colors)
        lab_color = imresize(lab_color, (args.imgSize, args.imgSize),
                             interp='nearest')

        # prediction
        pred_ = np.argmax(pred.data.cpu()[j].numpy(), axis=0)
        pred_color = colorEncode(pred_, colors)
        pred_color = imresize(pred_color, (args.imgSize, args.imgSize),
                              interp='nearest')

        # aggregate images and save
        im_vis = np.concatenate((img, lab_color, pred_color),
                                axis=1).astype(np.uint8)
        imsave(os.path.join(args.vis,
                            infos[j].replace('/', '_')
                            .replace('.jpg', '.png')), im_vis)


# train one epoch
def train(nets, loader, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    for net in nets:
        if not args.fix_bn:
            net.train()
        else:
            net.eval()

    # main loop
    tic = time.time()
    for i, batch_data in enumerate(loader):
        data_time.update(time.time() - tic)
        for net in nets:
            net.zero_grad()

        # forward pass
        pred, err = forward_with_loss(nets, batch_data, args, is_train=True)

        # Backward
        err.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # calculate accuracy, and display
        if i % args.disp_iter == 0:
            acc, _ = accuracy(batch_data, pred)

            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {}, lr_decoder: {}, '
                  'Accurarcy: {:4.2f}%, Loss: {}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_encoder, args.lr_decoder,
                          acc*100, err.data[0]))

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.data[0])
            history['train']['acc'].append(acc)


def evaluate(nets, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # switch to eval mode
    for net in nets:
        net.eval()

    for i, batch_data in enumerate(loader):
        # forward pass
        pred, err = forward_with_loss(nets, batch_data, args, is_train=False)
        loss_meter.update(err.data[0])
        print('[Eval] iter {}, loss: {}'.format(i, err.data[0]))

        # calculate accuracy
        acc, pix = accuracy(batch_data, pred)
        acc_meter.update(acc, pix)

        # visualization
        visualize(batch_data, pred, args)

    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['acc'].append(acc_meter.average())
    print('[Eval Summary] Epoch: {}, Loss: {}, Accurarcy: {:4.2f}%'
          .format(epoch, loss_meter.average(), acc_meter.average()*100))

    # Plot figure
    if epoch > 0:
        print('Plotting loss figure...')
        fig = plt.figure()
        plt.plot(np.asarray(history['train']['epoch']),
                 np.log(np.asarray(history['train']['err'])),
                 color='b', label='training')
        plt.plot(np.asarray(history['val']['epoch']),
                 np.log(np.asarray(history['val']['err'])),
                 color='c', label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Log(loss)')
        fig.savefig('{}/loss.png'.format(args.ckpt), dpi=200)
        plt.close('all')

        fig = plt.figure()
        plt.plot(history['train']['epoch'], history['train']['acc'],
                 color='b', label='training')
        plt.plot(history['val']['epoch'], history['val']['acc'],
                 color='c', label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        fig.savefig('{}/accuracy.png'.format(args.ckpt), dpi=200)
        plt.close('all')


def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_encoder, net_decoder, crit) = nets
    suffix_save = 'epoch{}.pth'.format(epoch)
    suffix_best = 'best.pth'

    if args.num_gpus > 1:
        dict_encoder = net_encoder.module.state_dict()
        dict_decoder = net_decoder.module.state_dict()
    else:
        dict_encoder = net_encoder.state_dict()
        dict_decoder = net_decoder.state_dict()

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_save))
    torch.save(dict_encoder,
               '{}/encoder_{}'.format(args.ckpt, suffix_save))
    torch.save(dict_decoder,
               '{}/decoder_{}'.format(args.ckpt, suffix_save))

    cur_err = history['val']['err'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(dict_encoder,
                   '{}/encoder_{}'.format(args.ckpt, suffix_best))
        torch.save(dict_decoder,
                   '{}/decoder_{}'.format(args.ckpt, suffix_best))


def create_optimizers(nets, args):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        net_encoder.parameters(),
        lr=args.lr_encoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        net_decoder.parameters(),
        lr=args.lr_decoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, epoch, args):
    drop_ratio = (1. * (args.num_epoch-epoch) / (args.num_epoch-epoch+1)) \
                 ** args.lr_pow
    args.lr_encoder *= drop_ratio
    args.lr_decoder *= drop_ratio
    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = args.lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = args.lr_decoder


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=args.arch_encoder,
                                        fc_dim=args.fc_dim,
                                        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(arch=args.arch_decoder,
                                        fc_dim=args.fc_dim,
                                        segSize=args.segSize,
                                        num_class=args.num_class,
                                        weights=args.weights_decoder)

    crit = nn.NLLLoss2d(ignore_index=-1)

    # Dataset and Loader
    dataset_train = Dataset(args.list_train, args, is_train=1)
    dataset_val = Dataset(args.list_val, args,
                          max_sample=args.num_val, is_train=0)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True)
    args.epoch_iters = int(len(dataset_train) / args.batch_size)
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # load nets into gpu
    if args.num_gpus > 1:
        net_encoder = nn.DataParallel(net_encoder,
                                      device_ids=range(args.num_gpus))
        net_decoder = nn.DataParallel(net_decoder,
                                      device_ids=range(args.num_gpus))
    nets = (net_encoder, net_decoder, crit)
    for net in nets:
        net.cuda()

    # Set up optimizers
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {split: {'epoch': [], 'err': [], 'acc': []}
               for split in ('train', 'val')}
    # initial eval
    evaluate(nets, loader_val, history, 0, args)
    for epoch in range(1, args.num_epoch + 1):
        train(nets, loader_train, optimizers, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(nets, loader_val, history, epoch, args)

        # checkpointing
        if epoch % args.ckpt_epoch == 0:
            checkpoint(nets, history, epoch, args)

        # adjust learning rate
        adjust_learning_rate(optimizers, epoch, args)

    print('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--arch_encoder', default='resnet34_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='c1_bilinear',
                        help="architecture of net_decoder")
    parser.add_argument('--weights_encoder', default='',
                        help="weights to finetune net_encoder")
    parser.add_argument('--weights_decoder', default='',
                        help="weights to finetune net_decoder")
    parser.add_argument('--fc_dim', default=512, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--list_train',
                        default='./data/ADE20K_object150_train.txt')
    parser.add_argument('--list_val',
                        default='./data/ADE20K_object150_val.txt')
    parser.add_argument('--root_img',
                        default='./data/ADEChallengeData2016/images')
    parser.add_argument('--root_seg',
                        default='./data/ADEChallengeData2016/annotations')

    # optimization related arguments
    parser.add_argument('--num_gpus', default=2, type=int,
                        help='number of gpus to use')
    parser.add_argument('--batch_size_per_gpu', default=16, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=100, type=int,
                        help='epochs to train for')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr_encoder', default=1e-3, type=float, help='LR')
    parser.add_argument('--lr_decoder', default=1e-2, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--fix_bn', default=0, type=int,
                        help='fix bn params')

    # Data related arguments
    parser.add_argument('--num_val', default=64, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers')
    parser.add_argument('--imgSize', default=384, type=int,
                        help='input image size')
    parser.add_argument('--segSize', default=384, type=int,
                        help='output image size')

    # Misc arguments
    parser.add_argument('--seed', default=1234, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--vis', default='./vis',
                        help='folder to output visualization during training')
    parser.add_argument('--disp_iter', type=int, default=20,
                        help='frequency to display')
    parser.add_argument('--eval_epoch', type=int, default=1,
                        help='frequency to evaluate')
    parser.add_argument('--ckpt_epoch', type=int, default=10,
                        help='frequency to checkpoint')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    if args.num_val < args.batch_size:
        args.num_val = args.batch_size

    args.id += '-' + str(args.arch_encoder)
    args.id += '-' + str(args.arch_decoder)
    args.id += '-ngpus' + str(args.num_gpus)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-imgSize' + str(args.imgSize)
    args.id += '-segSize' + str(args.segSize)
    args.id += '-lr_encoder' + str(args.lr_encoder)
    args.id += '-lr_decoder' + str(args.lr_decoder)
    args.id += '-epoch' + str(args.num_epoch)
    args.id += '-decay' + str(args.weight_decay)
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.vis, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    if not os.path.exists(args.vis):
        os.makedirs(args.vis)

    args.best_err = 2.e10   # initialize with a big number

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
