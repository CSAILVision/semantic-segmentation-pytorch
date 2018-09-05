# System libs
import os
import datetime
import glob
from scipy.io import loadmat
import cv2
import numpy as np

from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion

path_anno = './data/ADEChallengeData2016/annotations/validation'
result_semantic = '/data/vision/torralba/segmentation/semantic-segmentation-pytorch/result/baseline-resnet50_dilated8-ppm_bilinear_deepsup'
result_instance = '/data/vision/torralba/segmentation/Detectron-ADE/Outputs/ours_mask_rcnn_R-50-FPN_1x/Jul09-12-19-20_visiongpu15_step/test/vis_segmentation'

colors = loadmat('data/color150.mat')['colors']

ins2sem = {}
with open('/data/vision/torralba/segmentation/placeschallenge/instancesegmentation/categoryMapping.txt', 'r') as f:
    f.readline()
    for line in f:
        tokens = line.rstrip().split('\t')
        ins2sem[int(tokens[0])] = int(tokens[2])

print(ins2sem)

acc_meter = AverageMeter()
intersection_meter = AverageMeter()
union_meter = AverageMeter()

for file_pred in sorted(glob.glob(os.path.join(result_semantic, '*.png'))):
    pred_semantic = cv2.imread(file_pred, 0).astype(np.int)
    anno = cv2.imread(os.path.join(path_anno, os.path.basename(file_pred)), 0).astype(np.int)
    # print(pred.shape, anno.shape)

    # merge 
    file_instance = os.path.join(result_instance, os.path.basename(file_pred))
    if os.path.exists(file_instance):
        pred = pred_semantic
        pred_instance = cv2.imread(file_instance, 0).astype(np.int)
        #pred_instance -= 1
        #pred_instance = cv2.resize(pred_instance, (anno.shape[1], anno.shape[0]), cv2.INTER_NEAREST)
        for idx in np.unique(pred_instance):
            if idx < 1:
                continue
            mask = (pred_instance == idx)
            pred[mask] = ins2sem[idx]
        pred = pred - 1
    else:
        pred = pred_semantic - 1

    anno = anno - 1
    # calculate accuracy
    acc, pix = accuracy(pred, anno)
    intersection, union = intersectionAndUnion(pred, anno, 150)
    acc_meter.update(acc, pix)
    intersection_meter.update(intersection)
    union_meter.update(union)
    print('[{}], accuracy: {}'
          .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), acc))

iou = intersection_meter.sum / (union_meter.sum + 1e-10)
for i, _iou in enumerate(iou):
    print('class [{}], IoU: {}'.format(i, _iou))

print('[Eval Summary]:')
print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
      .format(iou.mean(), acc_meter.average()*100))

