#include <cstdio>
#include <iostream>

__global__ void factorial() {
    int a = 1;
    for(int i = 1; i <= threadIdx.x + 1; i++) {
        a *= i;
    }
    std::printf("%d!=%d\n", threadIdx.x + 1, a);
}

int main(void) {
    factorial<<<1,8>>>();
    cudaDeviceSynchronize();
    return 0;
}