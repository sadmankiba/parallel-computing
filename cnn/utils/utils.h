#ifndef UTILS_H
#define UTILS_H

void kaiming_init(float *weights, int nIn, int nOut);
void zeroes(float *arr, int n);
void ones(float *arr, int n);
void setEqual(float* src, float* dest, int size);
float arrMax(float* arr, int start, int end);

#endif