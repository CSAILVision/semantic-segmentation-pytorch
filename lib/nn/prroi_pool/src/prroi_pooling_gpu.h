/*
 * File   : prroi_pooling_gpu.h
 * Author : Jiayuan Mao, Tete Xiao
 * Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com 
 * Date   : 07/13/2018
 * 
 * Distributed under terms of the MIT license.
 * Copyright (c) 2017 Megvii Technology Limited.
 */

int prroi_pooling_forward_cuda(THCudaTensor *features, THCudaTensor *rois, THCudaTensor *output, int pooled_height, int pooled_width, float spatial_scale);

int prroi_pooling_backward_cuda(
    THCudaTensor *features, THCudaTensor *rois, THCudaTensor *output, THCudaTensor *output_diff, THCudaTensor *features_diff,
    int pooled_height, int pooled_width, float spatial_scale
);

int prroi_pooling_coor_backward_cuda(
    THCudaTensor *features, THCudaTensor *rois, THCudaTensor *output, THCudaTensor *output_diff, THCudaTensor *features_diff,
    int pooled_height, int pooled_width, float spatial_scal
);

