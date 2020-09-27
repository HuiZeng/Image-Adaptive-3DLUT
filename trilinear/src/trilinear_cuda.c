#include <THC/THC.h>
#include <math.h>
#include "trilinear_kernel.h"

extern THCState *state;

int trilinear_forward_cuda(THCudaTensor * lut, THCudaTensor * image, THCudaTensor * output,
                           int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * lut_flat = THCudaTensor_data(state, lut);
    float * image_flat = THCudaTensor_data(state, image);
    float * output_flat = THCudaTensor_data(state, output);

    // whether color image
    //int channels = THCudaTensor_size(state,image, 1);
    //if (channels != 3)
    //{
    //    return 0;
    //}

    cudaStream_t stream = THCState_getCurrentStream(state);

    TriLinearForwardLaucher(lut_flat, image_flat, output_flat, lut_dim, shift, binsize, width, height, batch, stream);

    return 1;
}

int trilinear_backward_cuda(THCudaTensor * image, THCudaTensor * image_grad, THCudaTensor * lut_grad,
                            int lut_dim, int shift, float binsize, int width, int height, int batch)
{
    // Grab the input tensor
    float * image_grad_flat = THCudaTensor_data(state, image_grad);
    float * image_flat = THCudaTensor_data(state, image);
    float * lut_grad_flat = THCudaTensor_data(state, lut_grad);

    // whether color image
    //int channels = THCudaTensor_size(state,image, 1);
    //if (channels != 3)
    //{
    //    return 0;
    //}

    cudaStream_t stream = THCState_getCurrentStream(state);
    TriLinearBackwardLaucher(image_flat, image_grad_flat, lut_grad_flat, lut_dim, shift, binsize, width, height, batch, stream);

    return 1;
}
