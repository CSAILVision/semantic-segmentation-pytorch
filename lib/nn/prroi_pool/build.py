#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : build.py
# Author : Jiayuan Mao, Tete Xiao
# Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
# Date   : 07/13/2018
# 
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

import os
import torch

from torch.utils.ffi import create_extension

headers = []
sources = []
defines = []
extra_objects = []
with_cuda = False

if torch.cuda.is_available():
    with_cuda = True

    headers+= ['src/prroi_pooling_gpu.h']
    sources += ['src/prroi_pooling_gpu.c']
    defines += [('WITH_CUDA', None)]

    this_file = os.path.dirname(os.path.realpath(__file__))
    extra_objects_cuda = ['src/prroi_pooling_gpu_impl.cu.o']
    extra_objects_cuda = [os.path.join(this_file, fname) for fname in extra_objects_cuda]
    extra_objects.extend(extra_objects_cuda)
else:
    # TODO(Jiayuan Mao @ 07/13): remove this restriction after we support the cpu implementation.
    raise NotImplementedError('Precise RoI Pooling only supports GPU (cuda) implememtations.')

ffi = create_extension(
    '_prroi_pooling',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()

