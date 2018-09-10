#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : prroi_pool.py
# Author : Jiayuan Mao, Tete Xiao
# Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
# Date   : 07/13/2018
# 
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

import torch.nn as nn

from .functional import prroi_pool2d

__all__ = ['PrRoIPool2D']


class PrRoIPool2D(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super().__init__()

        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return prroi_pool2d(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)
