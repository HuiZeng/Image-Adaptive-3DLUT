#include "trilinear_kernel.h"
#include <torch/extension.h>
#include <THC/THC.h>

int trilinear_forward_cuda(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                           int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * lut_flat = lut.data<float>();
    float * image_flat = image.data<float>();
    float * output_flat = output.data<float>();

    TriLinearForwardLaucher(lut_flat, image_flat, output_flat, lut_dim, shift, binsize, width, height, batch, at::cuda::getCurrentCUDAStream());

    return 1;
}

int trilinear_backward_cuda(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut_grad,
                            int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * image_grad_flat = image_grad.data<float>();
    float * image_flat = image.data<float>();
    float * lut_grad_flat = lut_grad.data<float>();

    TriLinearBackwardLaucher(image_flat, image_grad_flat, lut_grad_flat, lut_dim, shift, binsize, width, height, batch, at::cuda::getCurrentCUDAStream());

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &trilinear_forward_cuda, "Trilinear forward");
  m.def("backward", &trilinear_backward_cuda, "Trilinear backward");
}

