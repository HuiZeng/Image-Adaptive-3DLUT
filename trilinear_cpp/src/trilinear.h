#ifndef TRILINEAR_H
#define TRILINEAR_H

#include<torch/extension.h>

int trilinear_forward(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                      int lut_dim, int shift, float binsize, int width, int height, int batch);

int trilinear_backward(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch);

#endif
