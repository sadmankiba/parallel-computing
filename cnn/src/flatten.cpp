#include <vector>

#include "baselayer.h"
#include "flatten.h"

void init_weights() {}

Flatten::Flatten() {
    type = FLAT_TYPE;
}

V1D Flatten::forward(V3D input) {
    V1D output(input.size() * input[0].size() * input[0][0].size());
    for (unsigned int i = 0; i < input.size(); i++) {
        for (unsigned int j = 0; j < input[0].size(); j++) {
            for (unsigned int k = 0; k < input[0][0].size(); k++) {
                output[i * input[0].size() * input[0][0].size() + j * input[0][0].size() + k] = input[i][j][k];
            }
        }
    }
    return output;
}

V3D Flatten::calc_grad_z(V1D grad_z_nxt) {
    V3D grad_z_cur(in_size[0], 
        V2D(in_size[1], V1D(in_size[2])));
    for (int i = 0; i < in_size[0]; i++) {
        for (int j = 0; j < in_size[1]; j++) {
            for (int k = 0; k < in_size[2]; k++) {
                    grad_z_cur[i][j][k] = grad_z_nxt[i * in_size[0] * in_size[1] + j * in_size[1] + k];
            }
        }
    }

    return grad_z_cur;
}

void Flatten::update_weights() {
    // No weights to update
}

void Flatten::set_in_size(std::vector<int> _in_size) {
    in_size = _in_size;
}