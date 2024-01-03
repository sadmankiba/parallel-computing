#ifndef IMDATA_H
#define IMDATA_H

#include <vector>

/* 
A dataset of images. 
Each image can be of multiple channels.
Each channel is a 2D matrix of pixels.
Each image has a label of floating type.
*/
class ImageDataset {
private:
    int size;   /* dataset size */
    int chn; 
    int x;
    int y;
public:
    V4D images;
    V1D labels;

    // Constructor
    ImageDataset(unsigned int _size, int _chn, int _x, int _y): size(_size), chn(_chn), x(_x), y(_y) {}

    void addImage(V3D image, float label);

    void generate();

    void build(float* data, float* labels, int numData);
};

#endif