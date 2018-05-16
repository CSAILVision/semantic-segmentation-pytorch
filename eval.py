# System libs
import os
import datetime
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from dataset import ValDataset
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy, mark_volatile
import lib.utils.data as torchdata
import cv2


def visualize_result(data, preds, args):
    colors = loadmat('data/color150.mat')['colors']
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(preds, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    cv2.imwrite(os.path.join(args.result,
                img_name.replace('.jpg', '.png')), im_vis)


def evaluate(segmentation_module, loader, args):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    segmentation_module.eval()

    for i, batch_data in enumerate(loader):
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])

        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, args.gpu_id)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                pred = pred + pred_tmp.cpu() / len(args.imgSize)

            _, preds = torch.max(pred.data.cpu(), dim=1)
            preds = as_numpy(preds.squeeze(0))

        # calculate accuracy
        acc, pix = accuracy(preds, seg_label)
        intersection, union = intersectionAndUnion(preds, seg_label, args.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        print('[{}] iter {}, accuracy: {}'
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      i, acc))

        # visualization
        if args.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                preds, args)

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average()*100))


def main(args):
    torch.cuda.set_device(args.gpu_id)

    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_class,
        weights=args.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_val = ValDataset(
        args.list_val, args, max_sample=args.num_val)
    loader_val = torchdata.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, args)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', required=True,
                        help="a name for identifying the model to load")
    parser.add_argument('--suffix', default='_epoch_20.pth',
                        help="which snapshot to load")
    parser.add_argument('--arch_encoder', default='resnet50_dilated8',
                        help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup',
                        help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int,
                        help='number of features between encoder and decoder')

    # Path related arguments
    parser.add_argument('--list_val',
                        default='./data/validation.odgt')
    parser.add_argument('--root_dataset',
                        default='./data/')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--imgSize', default=[450], nargs='+', type=int,
                        help='list of input image sizes.'
                             'for multiscale testing, e.g.  300 400 500 600')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')

    # Misc arguments
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--visualize', action='store_true',
                        help='output visualization?')
    parser.add_argument('--result', default='./result',
                        help='folder to output visualization results')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='gpu_id for evaluation')

    args = parser.parse_args()
    print(args)

    # absolute paths of model weights
    args.weights_encoder = os.path.join(args.ckpt, args.id,
                                        'encoder' + args.suffix)
    args.weights_decoder = os.path.join(args.ckpt, args.id,
                                        'decoder' + args.suffix)
    assert os.path.exists(args.weights_encoder) and \
        os.path.exists(args.weights_encoder), 'checkpoint does not exitst!'

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
