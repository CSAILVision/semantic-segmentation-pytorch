/*
 * File   : prroi_pooling_gpu_impl.cu
 * Author : Tete Xiao, Jiayuan Mao
 * Email  : jasonhsiao97@gmail.com 
 * 
 * Distributed under terms of the MIT license.
 * Copyright (c) 2017 Megvii Technology Limited.
 */

#include "prroi_pooling_gpu_impl.cuh"

#include <cstdio>
#include <cfloat>

#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

#define CUDA_POST_KERNEL_CHECK \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (cudaSuccess != err) { \
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err)); \
            exit(-1); \
        } \
    } while(0)

#define CUDA_NUM_THREADS 512

namespace {

static int CUDA_NUM_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__device__ static float PrRoIPoolingGetData(F_DEVPTR_IN data, const int h, const int w, const int height, const int width)
{
    bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
    float retVal = overflow ? 0.0f : data[h * width + w];
    return retVal;
}

__device__ static float PrRoIPoolingGetCoeff(float dh, float dw){
    dw = dw > 0 ? dw : -dw;
    dh = dh > 0 ? dh : -dh;
    return (1.0f - dh) * (1.0f - dw);
}

__device__ static float PrRoIPoolingSingleCoorIntegral(float s, float t, float c1, float c2) {
    return 0.5 * (t * t - s * s) * c2 + (t - 0.5 * t * t - s + 0.5 * s * s) * c1;
}

__device__ static float PrRoIPoolingInterpolation(F_DEVPTR_IN data, const float h, const float w, const int height, const int width){
    float retVal = 0.0f;
    int h1 = floorf(h);
    int w1 = floorf(w);
    retVal += PrRoIPoolingGetData(data, h1, w1, height, width) * PrRoIPoolingGetCoeff(h - float(h1), w - float(w1));
    h1 = floorf(h)+1;
    w1 = floorf(w);
    retVal += PrRoIPoolingGetData(data, h1, w1, height, width) * PrRoIPoolingGetCoeff(h - float(h1), w - float(w1));
    h1 = floorf(h);
    w1 = floorf(w)+1;
    retVal += PrRoIPoolingGetData(data, h1, w1, height, width) * PrRoIPoolingGetCoeff(h - float(h1), w - float(w1));
    h1 = floorf(h)+1;
    w1 = floorf(w)+1;
    retVal += PrRoIPoolingGetData(data, h1, w1, height, width) * PrRoIPoolingGetCoeff(h - float(h1), w - float(w1));
    return retVal;
}

__device__ static float PrRoIPoolingMatCalculation(F_DEVPTR_IN this_data, const int s_h, const int s_w, const int e_h, const int e_w,
        const float y0, const float x0, const float y1, const float x1, const int h0, const int w0)
{
    float alpha, beta, lim_alpha, lim_beta, tmp;
    float sum_out = 0;

    alpha = x0 - float(s_w);
    beta = y0 - float(s_h);
    lim_alpha = x1 - float(s_w);
    lim_beta = y1 - float(s_h);
    tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
        * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
    sum_out += PrRoIPoolingGetData(this_data, s_h, s_w, h0, w0) * tmp;

    alpha = float(e_w) - x1;
    lim_alpha = float(e_w) - x0;
    tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
        * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
    sum_out += PrRoIPoolingGetData(this_data, s_h, e_w, h0, w0) * tmp;

    alpha = x0 - float(s_w);
    beta = float(e_h) - y1;
    lim_alpha = x1 - float(s_w);
    lim_beta = float(e_h) - y0;
    tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
        * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
    sum_out += PrRoIPoolingGetData(this_data, e_h, s_w, h0, w0) * tmp;

    alpha = float(e_w) - x1;
    lim_alpha = float(e_w) - x0;
    tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
        * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);   
    sum_out += PrRoIPoolingGetData(this_data, e_h, e_w, h0, w0) * tmp;

    return sum_out;
}

__device__ static void PrRoIPoolingDistributeDiff(F_DEVPTR_OUT diff, const float top_diff, const int h, const int w, const int height, const int width, const float coeff)
{
    bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
    if (!overflow) 
        atomicAdd(diff + h * width + w, top_diff * coeff);
}

__device__ static void PrRoIPoolingMatDistributeDiff(F_DEVPTR_OUT diff, const float top_diff, const int s_h, const int s_w, const int e_h, const int e_w,
        const float y0, const float x0, const float y1, const float x1, const int h0, const int w0)
{
    float alpha, beta, lim_alpha, lim_beta, tmp;

    alpha = x0 - float(s_w);
    beta = y0 - float(s_h);
    lim_alpha = x1 - float(s_w);
    lim_beta = y1 - float(s_h);
    tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
        * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
    PrRoIPoolingDistributeDiff(diff, top_diff, s_h, s_w, h0, w0, tmp);

    alpha = float(e_w) - x1;
    lim_alpha = float(e_w) - x0;
    tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
        * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
    PrRoIPoolingDistributeDiff(diff, top_diff, s_h, e_w, h0, w0, tmp);

    alpha = x0 - float(s_w);
    beta = float(e_h) - y1;
    lim_alpha = x1 - float(s_w);
    lim_beta = float(e_h) - y0;
    tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
        * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);
    PrRoIPoolingDistributeDiff(diff, top_diff, e_h, s_w, h0, w0, tmp);

    alpha = float(e_w) - x1;
    lim_alpha = float(e_w) - x0;
    tmp = (lim_alpha - 0.5f * lim_alpha * lim_alpha - alpha + 0.5f * alpha * alpha) 
        * (lim_beta - 0.5f * lim_beta * lim_beta - beta + 0.5f * beta * beta);   
    PrRoIPoolingDistributeDiff(diff, top_diff, e_h, e_w, h0, w0, tmp);
}

__global__ void PrRoIPoolingForward(
        const int nthreads, 
        F_DEVPTR_IN bottom_data,
        F_DEVPTR_IN bottom_rois,
        F_DEVPTR_OUT top_data,
        const int channels, 
        const int height,
        const int width, 
        const int pooled_height, 
        const int pooled_width,
        const float spatial_scale) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    
    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    
    float roi_start_w = bottom_rois[1] * spatial_scale;
    float roi_start_h = bottom_rois[2] * spatial_scale;
    float roi_end_w = bottom_rois[3] * spatial_scale;
    float roi_end_h = bottom_rois[4] * spatial_scale;

    float roi_width = max(roi_end_w - roi_start_w, ((float)0.0));
    float roi_height = max(roi_end_h - roi_start_h, ((float)0.0));
    float bin_size_h = roi_height / static_cast<float>(pooled_height);
    float bin_size_w = roi_width / static_cast<float>(pooled_width);

    const float *this_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
    float *this_out = top_data + index;

    float win_start_w = roi_start_w + bin_size_w * pw;
    float win_start_h = roi_start_h + bin_size_h * ph;
    float win_end_w = win_start_w + bin_size_w;
    float win_end_h = win_start_h + bin_size_h;
    
    float win_size = max(float(0.0), bin_size_w * bin_size_h);
    if (win_size == 0) {
        *this_out = 0;
        return;
    }

    float sum_out = 0;

    int s_w, s_h, e_w, e_h;
    
    s_w = floorf(win_start_w);
    e_w = ceilf(win_end_w);
    s_h = floorf(win_start_h);
    e_h = ceilf(win_end_h);

    for (int w_iter = s_w; w_iter < e_w; ++w_iter)
        for (int h_iter = s_h; h_iter < e_h; ++h_iter)
            sum_out += PrRoIPoolingMatCalculation(this_data, h_iter, w_iter, h_iter + 1, w_iter + 1, 
                max(win_start_h, float(h_iter)), max(win_start_w, float(w_iter)),
                min(win_end_h, float(h_iter) + 1.0), min(win_end_w, float(w_iter + 1.0)),
                height, width);
    *this_out = sum_out / win_size; 
  }
}

__global__ void PrRoIPoolingBackward(
        const int nthreads, 
        F_DEVPTR_IN bottom_rois,
        F_DEVPTR_IN top_diff,
        F_DEVPTR_OUT bottom_diff,
        const int channels, 
        const int height, 
        const int width,
        const int pooled_height, 
        const int pooled_width,
        const float spatial_scale) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    bottom_rois += n * 5; 
    
    int roi_batch_ind = bottom_rois[0];
    float roi_start_w = bottom_rois[1] * spatial_scale;
    float roi_start_h = bottom_rois[2] * spatial_scale;
    float roi_end_w = bottom_rois[3] * spatial_scale;
    float roi_end_h = bottom_rois[4] * spatial_scale;
    
    float roi_width = max(roi_end_w - roi_start_w, (float)0);
    float roi_height = max(roi_end_h - roi_start_h, (float)0);
    float bin_size_h = roi_height / static_cast<float>(pooled_height);
    float bin_size_w = roi_width / static_cast<float>(pooled_width);

    const float *this_out_grad = top_diff + index;
    float *this_data_grad = bottom_diff + (roi_batch_ind * channels + c) * height * width;

    float win_start_w = roi_start_w + bin_size_w * pw;
    float win_start_h = roi_start_h + bin_size_h * ph;
    float win_end_w = win_start_w + bin_size_w;
    float win_end_h = win_start_h + bin_size_h;

    float win_size = max(float(0.0), bin_size_w * bin_size_h);

    float sum_out = win_size == float(0) ? float(0) : *this_out_grad / win_size;

    int s_w, s_h, e_w, e_h;

    s_w = floorf(win_start_w);
    e_w = ceilf(win_end_w);
    s_h = floorf(win_start_h);
    e_h = ceilf(win_end_h);

    for (int w_iter = s_w; w_iter < e_w; ++w_iter)
        for (int h_iter = s_h; h_iter < e_h; ++h_iter)
            PrRoIPoolingMatDistributeDiff(this_data_grad, sum_out, h_iter, w_iter, h_iter + 1, w_iter + 1, 
                max(win_start_h, float(h_iter)), max(win_start_w, float(w_iter)),
                min(win_end_h, float(h_iter) + 1.0), min(win_end_w, float(w_iter + 1.0)),
                height, width);

  }
}

__global__ void PrRoIPoolingCoorBackward(
        const int nthreads, 
        F_DEVPTR_IN bottom_data,
        F_DEVPTR_IN bottom_rois,
        F_DEVPTR_IN top_data, 
        F_DEVPTR_IN top_diff,
        F_DEVPTR_OUT bottom_diff, 
        const int channels, 
        const int height, 
        const int width,
        const int pooled_height, 
        const int pooled_width,
        const float spatial_scale) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    bottom_rois += n * 5;

    int roi_batch_ind = bottom_rois[0];
    float roi_start_w = bottom_rois[1] * spatial_scale;
    float roi_start_h = bottom_rois[2] * spatial_scale;
    float roi_end_w = bottom_rois[3] * spatial_scale;
    float roi_end_h = bottom_rois[4] * spatial_scale;

    float roi_width = max(roi_end_w - roi_start_w, (float)0);
    float roi_height = max(roi_end_h - roi_start_h, (float)0);
    float bin_size_h = roi_height / static_cast<float>(pooled_height);
    float bin_size_w = roi_width / static_cast<float>(pooled_width);

    const float *this_out_grad = top_diff + index;
    const float *this_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
    const float *this_top_data = top_data + index;
    float *this_data_grad = bottom_diff + n * 5;
    
    float win_start_w = roi_start_w + bin_size_w * pw;
    float win_start_h = roi_start_h + bin_size_h * ph;
    float win_end_w = win_start_w + bin_size_w;
    float win_end_h = win_start_h + bin_size_h;

    float win_size = max(float(0.0), bin_size_w * bin_size_h);

    float sum_out = win_size == float(0) ? float(0) : *this_out_grad / win_size;
    
    // WARNING: to be discussed
    if (sum_out == 0)
        return;

    int s_w, s_h, e_w, e_h;

    s_w = floorf(win_start_w);
    e_w = ceilf(win_end_w);
    s_h = floorf(win_start_h);
    e_h = ceilf(win_end_h);

    float g_x1_y = 0, g_x2_y = 0, g_x_y1 = 0, g_x_y2 = 0;
    for (int h_iter = s_h; h_iter < e_h; ++h_iter) {
        g_x1_y += PrRoIPoolingSingleCoorIntegral(max(win_start_h, float(h_iter)) - h_iter, 
                min(win_end_h, float(h_iter + 1)) - h_iter, 
                PrRoIPoolingInterpolation(this_bottom_data, h_iter, win_start_w, height, width),
                PrRoIPoolingInterpolation(this_bottom_data, h_iter + 1, win_start_w, height, width));
    
        g_x2_y += PrRoIPoolingSingleCoorIntegral(max(win_start_h, float(h_iter)) - h_iter, 
                min(win_end_h, float(h_iter + 1)) - h_iter, 
                PrRoIPoolingInterpolation(this_bottom_data, h_iter, win_end_w, height, width),
                PrRoIPoolingInterpolation(this_bottom_data, h_iter + 1, win_end_w, height, width));
    }

    for (int w_iter = s_w; w_iter < e_w; ++w_iter) {
        g_x_y1 += PrRoIPoolingSingleCoorIntegral(max(win_start_w, float(w_iter)) - w_iter, 
                min(win_end_w, float(w_iter + 1)) - w_iter, 
                PrRoIPoolingInterpolation(this_bottom_data, win_start_h, w_iter, height, width),
                PrRoIPoolingInterpolation(this_bottom_data, win_start_h, w_iter + 1, height, width));
    
        g_x_y2 += PrRoIPoolingSingleCoorIntegral(max(win_start_w, float(w_iter)) - w_iter, 
                min(win_end_w, float(w_iter + 1)) - w_iter, 
                PrRoIPoolingInterpolation(this_bottom_data, win_end_h, w_iter, height, width),
                PrRoIPoolingInterpolation(this_bottom_data, win_end_h, w_iter + 1, height, width));
    }

    float partial_x1 = -g_x1_y + (win_end_h - win_start_h) * (*this_top_data);
    float partial_y1 = -g_x_y1 + (win_end_w - win_start_w) * (*this_top_data);
    float partial_x2 = g_x2_y - (win_end_h - win_start_h) * (*this_top_data);
    float partial_y2 = g_x_y2 - (win_end_w - win_start_w) * (*this_top_data);

    partial_x1 = partial_x1 / win_size * spatial_scale;
    partial_x2 = partial_x2 / win_size * spatial_scale;
    partial_y1 = partial_y1 / win_size * spatial_scale;
    partial_y2 = partial_y2 / win_size * spatial_scale;
    
    // (b, x1, y1, x2, y2)
    
    this_data_grad[0] = 0;
    atomicAdd(this_data_grad + 1, (partial_x1 * (1.0 - float(pw) / pooled_width) + partial_x2 * (1.0 - float(pw + 1) / pooled_width)) 
            * (*this_out_grad));
    atomicAdd(this_data_grad + 2, (partial_y1 * (1.0 - float(ph) / pooled_height) + partial_y2 * (1.0 - float(ph + 1) / pooled_height))
            * (*this_out_grad));
    atomicAdd(this_data_grad + 3, (partial_x2 * float(pw + 1) / pooled_width + partial_x1 * float(pw) / pooled_width)
            * (*this_out_grad)); 
    atomicAdd(this_data_grad + 4, (partial_y2 * float(ph + 1) / pooled_height + partial_y1 * float(ph) / pooled_height)
            * (*this_out_grad)); 
  }
}

} /* !anonymous namespace */

#ifdef __cplusplus
extern "C" {
#endif

void PrRoIPoolingForwardGpu(
    cudaStream_t stream,
    F_DEVPTR_IN bottom_data,
    F_DEVPTR_IN bottom_rois,
    F_DEVPTR_OUT top_data,
    const int channels_, const int height_, const int width_, 
    const int pooled_height_, const int pooled_width_,
    const float spatial_scale_,
    const int top_count) {

    PrRoIPoolingForward<<<CUDA_NUM_BLOCKS(top_count), CUDA_NUM_THREADS, 0, stream>>>(
        top_count, bottom_data, bottom_rois, top_data,
        channels_, height_, width_, pooled_height_, pooled_width_, spatial_scale_);

    CUDA_POST_KERNEL_CHECK;
}

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
    const int top_count, const int bottom_count) {

    cudaMemsetAsync(bottom_diff, 0, sizeof(float) * bottom_count, stream);
    PrRoIPoolingBackward<<<CUDA_NUM_BLOCKS(top_count), CUDA_NUM_THREADS, 0, stream>>>(
        top_count, bottom_rois, top_diff, bottom_diff,
        channels_, height_, width_, pooled_height_, pooled_width_, spatial_scale_);
    CUDA_POST_KERNEL_CHECK;
}

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
    const int top_count, const int bottom_count) {

    cudaMemsetAsync(bottom_diff, 0, sizeof(float) * bottom_count, stream);
    PrRoIPoolingCoorBackward<<<CUDA_NUM_BLOCKS(top_count), CUDA_NUM_THREADS, 0, stream>>>(
        top_count, bottom_data, bottom_rois, top_data, top_diff, bottom_diff,
        channels_, height_, width_, pooled_height_, pooled_width_, spatial_scale_);
    CUDA_POST_KERNEL_CHECK;
}

} /* !extern "C" */

