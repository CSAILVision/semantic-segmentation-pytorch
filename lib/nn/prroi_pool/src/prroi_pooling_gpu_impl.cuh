/*
 * File   : prroi_pooling_gpu_impl.cuh
 * Author : Tete Xiao, Jiayuan Mao
 * Email  : jasonhsiao97@gmail.com 
 * 
 * Distributed under terms of the MIT license.
 * Copyright (c) 2017 Megvii Technology Limited.
 */

#ifndef PRROI_POOLING_GPU_IMPL_CUH
#define PRROI_POOLING_GPU_IMPL_CUH

#ifdef __cplusplus
extern "C" {
#endif

#define F_DEVPTR_IN const float * 
#define F_DEVPTR_OUT float * 

void PrRoIPoolingForwardGpu(
    cudaStream_t stream,
    F_DEVPTR_IN bottom_data,
    F_DEVPTR_IN bottom_rois,
    F_DEVPTR_OUT top_data,
    const int channels_, const int height_, const int width_, 
    const int pooled_height_, const int pooled_width_,
    const float spatial_scale_,
    const int top_count);

void PrRoIPoolingBackwardGpu(
    cudaStream_t stream,
    F_DEVPTR_IN bottom_data,
    F_DEVPTR_IN bottom_rois,
    F_DEVPTR_IN top_data,
    F_DEVPTR_IN top_diff,
    F_DEVPTR_OUT bottom_diff,
    const int channels_, const int height_, const int width_, 
    const int pooled_height_, const int pooled_width_, 
    const float spatial_scale_,
    const int top_count, const int bottom_count);

void PrRoIPoolingCoorBackwardGpu(
    cudaStream_t stream,
    F_DEVPTR_IN bottom_data,
    F_DEVPTR_IN bottom_rois,
    F_DEVPTR_IN top_data,
    F_DEVPTR_IN top_diff,
    F_DEVPTR_OUT bottom_diff,
    const int channels_, const int height_, const int width_, 
    const int pooled_height_, const int pooled_width_, 
    const float spatial_scale_,
    const int top_count, const int bottom_count);

#ifdef __cplusplus
} /* !extern "C" */
#endif

#endif /* !PRROI_POOLING_GPU_IMPL_CUH */

