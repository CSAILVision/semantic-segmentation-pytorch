#! /bin/bash -e
# File   : travis.sh
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
#
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

cd src
echo "Working directory: " `pwd`
echo "Compiling prroi_pooling kernels by nvcc..."
nvcc -c -o prroi_pooling_gpu_impl.cu.o prroi_pooling_gpu_impl.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
echo "Working directory: " `pwd`
echo "Building python libraries..."
python3 build.py

