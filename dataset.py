import os
import random
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
from scipy.misc import imread, imresize


class Dataset(torchdata.Dataset):
    def __init__(self, txt, opt, max_sample=-1, is_train=1):
        self.root_img = opt.root_img
        self.root_seg = opt.root_seg
        self.imgSize = opt.imgSize
        self.segSize = opt.segSize
        self.is_train = is_train

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.list_sample = [x.rstrip() for x in open(txt, 'r')]

        if self.is_train:
            random.shuffle(self.list_sample)
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def _scale_and_crop(self, img, seg, cropSize, is_train):
        h, w = img.shape[0], img.shape[1]

        if is_train:
            # random scale
            scale = random.random() + 0.5     # 0.5-1.5
            scale = max(scale, 1. * cropSize / (min(h, w) - 1))
        else:
            # scale to crop size
            scale = 1. * cropSize / (min(h, w) - 1)

        img_scale = imresize(img, scale, interp='bilinear')
        seg_scale = imresize(seg, scale, interp='nearest')

        h_s, w_s = img_scale.shape[0], img_scale.shape[1]
        if is_train:
            # random crop
            x1 = random.randint(0, w_s - cropSize)
            y1 = random.randint(0, h_s - cropSize)
        else:
            # center crop
            x1 = (w_s - cropSize) // 2
            y1 = (h_s - cropSize) // 2

        img_crop = img_scale[y1: y1 + cropSize, x1: x1 + cropSize, :]
        seg_crop = seg_scale[y1: y1 + cropSize, x1: x1 + cropSize]
        return img_crop, seg_crop

    def _flip(self, img, seg):
        img_flip = img[:, ::-1, :]
        seg_flip = seg[:, ::-1]
        return img_flip, seg_flip

    def __getitem__(self, index):
        img_basename = self.list_sample[index]
        path_img = os.path.join(self.root_img, img_basename)
        path_seg = os.path.join(self.root_seg,
                                img_basename.replace('.jpg', '.png'))

        assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)
        assert os.path.exists(path_seg), '[{}] does not exist'.format(path_seg)

        # load image and label
        try:
            img = imread(path_img, mode='RGB')
            seg = imread(path_seg)
            assert(img.ndim == 3)
            assert(seg.ndim == 2)
            assert(img.shape[0] == seg.shape[0])
            assert(img.shape[1] == seg.shape[1])

            # random scale, crop, flip
            if self.imgSize > 0:
                img, seg = self._scale_and_crop(img, seg,
                                                self.imgSize, self.is_train)
                if random.choice([-1, 1]) > 0:
                    img, seg = self._flip(img, seg)

            # image to float
            img = img.astype(np.float32) / 255.
            img = img.transpose((2, 0, 1))

            if self.segSize > 0:
                seg = imresize(seg, (self.segSize, self.segSize),
                               interp='nearest')

            # label to int from -1 to 149
            seg = seg.astype(np.int) - 1

            # to torch tensor
            image = torch.from_numpy(img)
            segmentation = torch.from_numpy(seg)
        except Exception as e:
            print('Failed loading image/segmentation [{}]: {}'
                  .format(path_img, e))
            # dummy data
            image = torch.zeros(3, self.imgSize, self.imgSize)
            segmentation = -1 * torch.ones(self.segSize, self.segSize).long()
            return image, segmentation, img_basename

        # substracted by mean and divided by std
        image = self.img_transform(image)

        return image, segmentation, img_basename

    def __len__(self):
        return len(self.list_sample)
