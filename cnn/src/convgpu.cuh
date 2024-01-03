#ifndef CONVGPU_H
#define CONVGPU_H

int conv_gpu(float * input_h,float * filter_h, float * output_h, unsigned int input_size, unsigned int filter_size, char *method);

#endif // CONVGPU_H
