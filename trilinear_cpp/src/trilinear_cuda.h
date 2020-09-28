#ifndef TRILINEAR_CUDA_H
#define TRILINEAR_CUDA_H

#import <torch/extension.h>

int trilinear_forward_cuda(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                           int lut_dim, int shift, float binsize, int width, int height, int batch);

int trilinear_backward_cuda(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                            int lut_dim, int shift, float binsize, int width, int height, int batch);

#endif
