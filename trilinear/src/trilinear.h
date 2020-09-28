int trilinear_forward(THFloatTensor * lut, THFloatTensor * image, THFloatTensor * output,
                      int lut_dim, int shift, float binsize, int width, int height, int batch);

int trilinear_backward(THFloatTensor * image, THFloatTensor * image_grad, THFloatTensor * lut_grad,
                       int lut_dim, int shift, float binsize, int width, int height, int batch);

