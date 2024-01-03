#include <vector>
#include <iostream>

#include "../src/baselayer.h"
#include "../src/mdvector.h"
#include "../src/imdata.h"

/* 
A dataset of images. 
Each image can be of multiple channels.
Each channel is a 2D matrix of pixels.
Each image has a label of floating type.
*/
    
// Add an image to the dataset
void ImageDataset::addImage(V3D image, float label) {
    images.push_back(image);
    labels.push_back(label);
}

/* 
Make a dataset of two classes 
Alg: 
Each 2D channel is a regularly increasing value between 0 to 1. 
Actually, ith image has values between 0 to (i / size).
So, each data point (image) has larger range than previous in dataset.
The channels in an image have same values but different signs.
First half of the dataset has label 0, second half has label 1.
*/
void ImageDataset::generate() {
    for (int i = 0; i < size; i++) {
        V3D image(chn, V2D(x, V1D(y)));
        for (int c = 0; c < chn; c++){ 
            for (int j = 0; j < x; j++) {
                for (int k = 0; k < y; k++) {
                    image[c][j][k] = 1.0f * ((c % 2) * 2 - 1) * (i + 1) * (j * x + k) / (x * y * size);
                }
            }
        }
        addImage(image, (int) ((i * 2) / size));
    }

    // Print the number of images in the dataset
    std::cout << "Number of images in the dataset: " << this->images.size() << std::endl;

    // Print the first image in the dataset
    V3D firstImage = this->images[0];
    std::cout << "1st image, channel 0:" << std::endl;
    for (const auto& row : firstImage[0]) {
        for (const auto& pixel : row) {
            std::cout << pixel << " ";
        }
        std::cout << std::endl;
    }   
}

void ImageDataset::build(float* data, float* labels, int numData) {
    for (int i = 0; i < numData; i++) {
        V3D image(chn, V2D(x, V1D(y)));
        for (int c = 0; c < chn; c++){ 
            for (int j = 0; j < x; j++) {
                for (int k = 0; k < y; k++) {
                    image[c][j][k] = data[i * chn * x * y + c * x * y + j * y + k];
                }
            }
        }
        addImage(image, labels[i]);
    }
}
