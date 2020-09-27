int trilinear_forward(THCudaTensor * lut, THCudaTensor * image, THCudaTensor * output,
                      int lut_dim, int shift, float binsize, int width, int height, int batch);

int trilinear_backward(THCudaTensor * image, THCudaTensor * image_grad, THCudaTensor * lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch);
