#include <cstddef>
#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    for (int p=0; p<n; p++){
        for (int q=0; q<n; q++){
            float item = 0;
            for (int i=0; i<m; i++){
                for (int j=0; j<m; j++){
                    // calculate coordinates to access in image
                    int xImg = p+i-(m-1)/2;
                    int yImg = q+j-(m-1)/2;

                    float itemImg;
                    // set itemImg based on boundaries
                    if ((xImg<0 || xImg>=n) && (yImg<0 || yImg>=n)){
                        itemImg = 0;
                    } else if (xImg<0 || xImg>=n) {
                        itemImg = 1;
                    } else if (yImg<0 || yImg>=n) {
                        itemImg = 1;
                    } else {
                        itemImg = image[xImg*n + yImg];
                    }

                    item += mask[i*m + j] * itemImg;
                }
            }
            output[p*n + q] = item;
        }
    }
    return;
}