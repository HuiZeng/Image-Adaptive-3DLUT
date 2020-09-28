#include <math.h>
#include <float.h>
#include "trilinear_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


__global__ void TriLinearForward(const int nthreads, const float* lut, const float* image, float* output, const int dim, const int shift, const float binsize, const int width, const int height, const int batch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

        float r = image[index];
	float g = image[index + width * height * batch];
	float b = image[index + width * height * batch * 2];

	int r_id = floor(r / binsize);
	int g_id = floor(g / binsize);
	int b_id = floor(b / binsize);

        float r_d = fmod(r,binsize) / binsize;
        float g_d = fmod(g,binsize) / binsize;
        float b_d = fmod(b,binsize) / binsize;

        int id000 = r_id + g_id * dim + b_id * dim * dim;
        int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
        int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
        int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
        int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
        int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
        int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
        int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;

        float w000 = (1-r_d)*(1-g_d)*(1-b_d);
        float w100 = r_d*(1-g_d)*(1-b_d);
        float w010 = (1-r_d)*g_d*(1-b_d);
        float w110 = r_d*g_d*(1-b_d);
        float w001 = (1-r_d)*(1-g_d)*b_d;
        float w101 = r_d*(1-g_d)*b_d;
        float w011 = (1-r_d)*g_d*b_d;
        float w111 = r_d*g_d*b_d;

        output[index] = w000 * lut[id000] + w100 * lut[id100] + 
                        w010 * lut[id010] + w110 * lut[id110] + 
                        w001 * lut[id001] + w101 * lut[id101] + 
                        w011 * lut[id011] + w111 * lut[id111];

        output[index + width * height * batch] = w000 * lut[id000 + shift] + w100 * lut[id100 + shift] + 
                                                 w010 * lut[id010 + shift] + w110 * lut[id110 + shift] + 
                                                 w001 * lut[id001 + shift] + w101 * lut[id101 + shift] + 
                                                 w011 * lut[id011 + shift] + w111 * lut[id111 + shift];

        output[index + width * height * batch * 2] = w000 * lut[id000 + shift * 2] + w100 * lut[id100 + shift * 2] + 
                                                     w010 * lut[id010 + shift * 2] + w110 * lut[id110 + shift * 2] + 
                                                     w001 * lut[id001 + shift * 2] + w101 * lut[id101 + shift * 2] + 
                                                     w011 * lut[id011 + shift * 2] + w111 * lut[id111 + shift * 2];

    }
}


int TriLinearForwardLaucher(const float* lut, const float* image, float* output, const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream) {
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;
    cudaError_t err;


    TriLinearForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, lut, image, output, lut_dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


__global__ void TriLinearBackward(const int nthreads, const float* image, const float* image_grad, float* lut_grad, const int dim, const int shift, const float binsize, const int width, const int height, const int batch) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

    float r = image[index];
    float g = image[index + width * height * batch];
    float b = image[index + width * height * batch * 2];

    int r_id = floor(r / binsize);
    int g_id = floor(g / binsize);
    int b_id = floor(b / binsize);

    float r_d = fmod(r,binsize) / binsize;
    float g_d = fmod(g,binsize) / binsize;
    float b_d = fmod(b,binsize) / binsize;

    int id000 = r_id + g_id * dim + b_id * dim * dim;
    int id100 = r_id + 1 + g_id * dim + b_id * dim * dim;
    int id010 = r_id + (g_id + 1) * dim + b_id * dim * dim;
    int id110 = r_id + 1 + (g_id + 1) * dim + b_id * dim * dim;
    int id001 = r_id + g_id * dim + (b_id + 1) * dim * dim;
    int id101 = r_id + 1 + g_id * dim + (b_id + 1) * dim * dim;
    int id011 = r_id + (g_id + 1) * dim + (b_id + 1) * dim * dim;
    int id111 = r_id + 1 + (g_id + 1) * dim + (b_id + 1) * dim * dim;

    float w000 = (1-r_d)*(1-g_d)*(1-b_d);
    float w100 = r_d*(1-g_d)*(1-b_d);
    float w010 = (1-r_d)*g_d*(1-b_d);
    float w110 = r_d*g_d*(1-b_d);
    float w001 = (1-r_d)*(1-g_d)*b_d;
    float w101 = r_d*(1-g_d)*b_d;
    float w011 = (1-r_d)*g_d*b_d;
    float w111 = r_d*g_d*b_d;

    atomicAdd(lut_grad + id000, image_grad[index] * w000);
    atomicAdd(lut_grad + id100, image_grad[index] * w100);
    atomicAdd(lut_grad + id010, image_grad[index] * w010);
    atomicAdd(lut_grad + id110, image_grad[index] * w110);
    atomicAdd(lut_grad + id001, image_grad[index] * w001);
    atomicAdd(lut_grad + id101, image_grad[index] * w101);
    atomicAdd(lut_grad + id011, image_grad[index] * w011);
    atomicAdd(lut_grad + id111, image_grad[index] * w111);

    atomicAdd(lut_grad + id000 + shift, image_grad[index + width * height * batch] * w000);
    atomicAdd(lut_grad + id100 + shift, image_grad[index + width * height * batch] * w100);
    atomicAdd(lut_grad + id010 + shift, image_grad[index + width * height * batch] * w010);
    atomicAdd(lut_grad + id110 + shift, image_grad[index + width * height * batch] * w110);
    atomicAdd(lut_grad + id001 + shift, image_grad[index + width * height * batch] * w001);
    atomicAdd(lut_grad + id101 + shift, image_grad[index + width * height * batch] * w101);
    atomicAdd(lut_grad + id011 + shift, image_grad[index + width * height * batch] * w011);
    atomicAdd(lut_grad + id111 + shift, image_grad[index + width * height * batch] * w111);

    atomicAdd(lut_grad + id000 + shift * 2, image_grad[index + width * height * batch * 2] * w000);
    atomicAdd(lut_grad + id100 + shift * 2, image_grad[index + width * height * batch * 2] * w100);
    atomicAdd(lut_grad + id010 + shift * 2, image_grad[index + width * height * batch * 2] * w010);
    atomicAdd(lut_grad + id110 + shift * 2, image_grad[index + width * height * batch * 2] * w110);
    atomicAdd(lut_grad + id001 + shift * 2, image_grad[index + width * height * batch * 2] * w001);
    atomicAdd(lut_grad + id101 + shift * 2, image_grad[index + width * height * batch * 2] * w101);
    atomicAdd(lut_grad + id011 + shift * 2, image_grad[index + width * height * batch * 2] * w011);
    atomicAdd(lut_grad + id111 + shift * 2, image_grad[index + width * height * batch * 2] * w111);
}
    }

int TriLinearBackwardLaucher(const float* image, const float* image_grad, float* lut_grad, const int lut_dim, const int shift, const float binsize, const int width, const int height, const int batch, cudaStream_t stream) {
    const int kThreadsPerBlock = 1024;
    const int output_size = height * width * batch;
    cudaError_t err;

    TriLinearBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, image, image_grad, lut_grad, lut_dim, shift, binsize, width, height, batch);

    err = cudaGetLastError();
    if(cudaSuccess != err) {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}
