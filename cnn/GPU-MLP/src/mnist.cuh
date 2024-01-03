#include <string>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

class MNIST{
    public:
        MNIST();
        float* readData(std::string filepath, int numData = -1);
        float* readLabels(std::string filepath, int numData = -1);
};

void oneHotEncodeLabels(float* labels, thrust::device_vector<float> &labelsVector, int numData, int numClasses);