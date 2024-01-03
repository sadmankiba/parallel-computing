#include "../src/linear.cuh"
#include "../src/relu.cuh"
#include "../src/softmax.cuh"
#include "../src/nllloss.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>

using namespace std;

// implemented using Copilot

// Function to print the contents of a thrust::device_vector
void device_vector_cout(const thrust::device_vector<float>& d_vec, const string& name = "") {
    cout << name << ": ";
    thrust::copy(d_vec.begin(), d_vec.end(), ostream_iterator<float>(cout, ", "));
    cout << endl;
}

// Function to compare a thrust::device_vector with a vector
bool device_vector_equals_vector(const thrust::device_vector<float>& d_vec, const vector<float>& vec) {
    if (d_vec.size() != vec.size()) {
        return false;
    }

    for (size_t i = 0; i < d_vec.size(); ++i) {
        if (abs(d_vec[i] - vec[i]) > FLT_EPSILON) {
            return false;
        }
    }

    return true;
}


int main(int argc, char **argv) {
    Linear layer(4,5,0.1f,0.99f);
    vector<float> input_host = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    thrust::device_vector<float> input = input_host;
    vector<float> weights(20, 1.0f);
    layer.weights = weights;

    // test forward
    cout << "test forward" << endl;
	device_vector_cout(input, "input");
	device_vector_cout(layer.weights, "weights");
    layer.forward(input,2);
    device_vector_cout(layer.output, "output");
    cout << endl;

    // test update
    cout << "test update" << endl;
    device_vector_cout(layer.bias, "init bias");
	device_vector_cout(layer.biasSquareGrad, "init biasSquareGrad");
	device_vector_cout(layer.wtSquareGrad, "init wtSquareGrad");
    thrust::device_vector<float> grad(10, 1.0f);
    layer.update(grad, 2);
    device_vector_cout(layer.bias, "updated bias");
	device_vector_cout(layer.biasSquareGrad, "updated biasSquareGrad");
    device_vector_cout(layer.wtSquareGrad, "updated wtSquareGrad");
    device_vector_cout(layer.newWeights, "updated weights");
    cout << endl;

    // test backward
    cout << "test backward" << endl;
    layer.backward(grad, 2);
    device_vector_cout(layer.weights, "weights");
    device_vector_cout(layer.gradient, "gradient");
    cout << endl;

    // test relu
    cout << "test relu" << endl;
    ReLU relu(4);
    input_host[1] = 0.0f;
    input_host[4] = 0.0f;
    // {1, 0, 3, 4, 0, 6, 7, 8}
    input = input_host;
    relu.forward(input, 2);
    vector<float> expOp = {1.0f, 0.0f, 3.0f, 4.0f, 0.0f, 6.0f, 7.0f, 8.0f};
    bool match = device_vector_equals_vector(relu.output, expOp);
    cout << "relu output match: " << match << endl;
    relu.backward(grad, 2);
    expOp = {1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    match = device_vector_equals_vector(relu.gradient, expOp);
    cout << "relu gradient match: " << match << endl;
    cout << endl;

    // test softmax
    cout << "test softmax" << endl;
    Softmax softmax(4);
    // {1, 0, 3, 4, 0, 6, 7, 8}
    softmax.forward(input, 2);
    device_vector_cout(softmax.output, "softmax output");
    softmax.backward(grad, 2);
    device_vector_cout(softmax.gradient, "softmax gradient");
    cout << endl;

    // test loss
    cout << "test loss" << endl;
    NLLLoss loss(4);
    vector<float> labels = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    thrust::device_vector<float> labelsD = labels;
    loss.forward(input, labelsD, 2);
    cout << "loss: " << loss.loss << endl;
    loss.backward(2);
    device_vector_cout(loss.gradient, "loss gradient");
    cout << endl;
}