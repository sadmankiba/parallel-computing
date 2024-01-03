#include <vector>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cuda.h>

#include "baselayer.h"
#include "convgpu.cuh"
#include "conv.h"

Conv::Conv(int _chn_prev, int _chn_cur, int _fs, bool _use_cpu) : 
        fs(_fs), chn_prev(_chn_prev), chn_cur(_chn_cur),
        weights(_chn_cur, V2D(_fs, V1D(_fs))), bias(_chn_cur), conv_time(0) {
    init_weights();
    init_bias();
    use_cpu = _use_cpu;
    type = CONV_TYPE;
    size_cur.resize(3); /* dim 1 - cur channel, dim 2 - cur img x, dim 3 - cur img y */
    size_cur[0] = chn_cur; 
}

V3D Conv::forward(V3D input) {
    input_last = input;
    V3D z(size_cur[0], V2D(size_cur[1], V1D(size_cur[2])));
    V3D a(size_cur[0], V2D(size_cur[1], V1D(size_cur[2])));
    
    for (int i = 0; i < chn_cur; i++) {
        z[i] = conv(input[0], weights[i]); /* Assuming input is of 1-channel */
        for (int j = 0; j < size_cur[1]; j++) {
            for (int k = 0; k < size_cur[2]; k++) {
                z[i][j][k] += bias[i];
            }
        }
        for (unsigned int j = 0; j < z[0].size(); j++) {
            for (unsigned int k = 0; k < z[0][0].size(); k++) {
                a[i][j][k] = activ_forward(z[i][j][k]);
            }
        }
    }
    a_last = a;
    return a;
}

V3D Conv::calc_grad_z(V3D w_nxt, V3D grad_z_nxt) {
    assert(grad_z_nxt.size() == size_cur[0] - w_nxt.size() + 1);
    assert(grad_z_nxt[0].size() == size_cur[1] - w_nxt[0].size() + 1);
    assert(grad_z_nxt[0][0].size() == size_cur[2] - w_nxt[0][0].size() + 1);

    V2D grad_a_cur(size_cur[1], V1D(size_cur[2]));
    for (int i = 0; i < size_cur[0]; i++) {
        grad_a_cur = conv(
            pad(
                rot_180(w_nxt[i]), 
                grad_z_nxt.size() - 1,
                grad_z_nxt[0].size() - 1 
            ), 
            grad_z_nxt[i]
        );
        for (int j = 0; j < size_cur[1]; j++) {
            for (int k = 0; k < size_cur[2]; k++) {
                grad_z_cur[i][j][k] = grad_a_cur[j][k] * activ_grad_from_a(a_last[i][j][k]);
            }
        }
    }

    return grad_z_cur;
}

void Conv::update_weights() {
    V2D dW(weights[0].size(), V1D(weights[0][0].size()));
    V1D db(bias.size());
    for (unsigned int i = 0; i < weights.size(); i++) { /* iterating over cur channels */
        dW = conv(input_last[0], grad_z_cur[i]);
        /* 
        Good Breakpoint 
        Check grad_z_cur, dW, weights 
        */
        for (unsigned int j = 0; j < weights[0].size(); j++) {
            for (unsigned int k = 0; k < weights[0][0].size(); k++) {
                weights[i][j][k] -= lr * dW[j][k];
            }
        }

        for (unsigned int j = 0; j < grad_z_cur[0].size(); j++) {
            for (unsigned int k = 0; k < grad_z_cur[0][0].size(); k++) {
                db[i] += grad_z_cur[i][j][k];
            }
        }
        db[i] /= grad_z_cur[0].size() * grad_z_cur[0][0].size();
        bias[i] -= lr * db[i];
    }
}

void Conv::set_in_size(std::vector<int> _in_size) {
    size_cur[1] = _in_size[1] - fs + 1;
    size_cur[2] = _in_size[2] - fs + 1;
    grad_z_cur.resize(size_cur[0], V2D(size_cur[1], V1D(size_cur[2])));
} 

void Conv::set_grad_z(V3D grad_z) {
    grad_z_cur = grad_z;
}

/* 
Initialize weights randomly 
Between -1 to 1
*/
void Conv::init_weights() {
    V3D w(chn_cur, V2D(fs, V1D(fs)));
    for (int i = 0; i < chn_cur; i++) {
        for (int j = 0; j < fs; j++) {
            for (int k = 0; k < fs; k++) {
                w[i][j][k] = ((float)rand() / RAND_MAX) * 2.0 - 1;
            }
        }
    }
    weights = w;
}

/* 
Initialize bias randomly 
Between -1 to 1
*/
void Conv::init_bias() {
    for (int i = 0; i < chn_cur; i++) {
        bias[i] = ((float) rand() / RAND_MAX) * 2.0 - 1;
    }
}

V2D conv_cpu(V2D input, V2D filter) {
    V2D output(input.size() - filter.size() + 1, V1D(input[0].size() - filter[0].size() + 1));
    for (unsigned int i = 0; i < input.size() - filter.size() + 1; i++) {
        for (unsigned int j = 0; j < input[0].size() - filter[0].size() + 1; j++) {
            float sum = 0;
            for (unsigned int k = 0; k < filter.size(); k++) {
                for (unsigned int l = 0; l < filter[0].size(); l++) {
                    sum += input[i + k][j + l] * filter[k][l];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

int conv_cpu_fp(float* input, float* filter, float* output, unsigned int input_size, unsigned int filter_size) {
    unsigned int output_size = input_size - filter_size + 1;

    for (unsigned int i = 0; i < output_size; i++) {
        for (unsigned int j = 0; j < output_size; j++) {
            float sum = 0;
            for (unsigned int k = 0; k < filter_size; k++) {
                for (unsigned int l = 0; l < filter_size; l++) {
                    sum += input[(i + k) * input_size + (j + l)] * filter[k * filter_size + l];
                }
            }
            output[i * output_size + j] = sum;
        }
    }
    return 0;
}

V2D Conv::conv(V2D input_vec, V2D filter_vec) {
    assert(input_vec.size() == input_vec[0].size());
    assert(filter_vec.size() == filter_vec[0].size());
    
    float *input, *filter, *output;

    unsigned int input_size = input_vec.size();
    unsigned int filter_size = filter_vec.size();
    int output_size = input_size - filter_size + 1;

    V2D output_vec(output_size, V1D(output_size));

    input = (float *)malloc(input_size * input_size * sizeof(float));
    filter = (float *)malloc(filter_size * filter_size * sizeof(float));
    output = (float *)malloc(output_size * output_size * sizeof(float));

    for (int i = 0; i < input_size * input_size; i++) {
        input[i] = input_vec[i / input_size][i % input_size];
    }

    for (int i = 0; i < filter_size * filter_size; i++) {
        filter[i] = filter_vec[i / filter_size][i % filter_size];
    }

    auto conv_start = std::chrono::high_resolution_clock::now();
    if (this->use_cpu) {
        conv_cpu_fp(input, filter, output, input_size, filter_size);
    }
    else {
	char method[20] = "gpu-sm";
        conv_gpu(input, filter, output, input_size, filter_size, method);
    }
    auto conv_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(conv_end - conv_start).count();
    this->conv_time += duration;

    for(int i = 0; i < output_size * output_size; i++) {
        output_vec[i / output_size][i % output_size] = output[i];
    }


    free(input);
    free(filter);
    free(output);

    return output_vec;

    
}

V2D Conv::rot_180(const V2D& input) {
    int rows = input.size();
    int cols = input[0].size();
    V2D rotated(rows, V1D(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            rotated[i][j] = input[rows - 1 - i][cols - 1 - j];
        }
    }

    return rotated;
}

V2D Conv::pad(const V2D& input, int pad_rows, int pad_cols) {
    int rows = input.size();
    int cols = input[0].size();
    V2D padded(rows + 2 * pad_rows, V1D(cols + 2 * pad_cols, 0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            padded[i + pad_rows][j + pad_cols] = input[i][j];
        }
    }

    return padded;
}

void Conv::print_weights() {
    std::cout << "Conv weights: " << std::endl;
    for (int i = 0; i < chn_cur; i++) {
        std::cout << "Channel " << i << std::endl;
        for (int j = 0; j < fs; j++) {
            for (int k = 0; k < fs; k++) {
                std::cout << weights[i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Bias: " << bias[i] << std::endl;
    }
}
