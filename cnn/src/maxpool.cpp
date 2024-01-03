#include <vector>

#include "baselayer.h"
#include "maxpool.h"

MaxPool::MaxPool(int fs) : fs(fs) {
    type = POOL_TYPE;
}

V3D MaxPool::forward(V3D input) {
    V3D output(input.size(), V2D(input[0].size() / fs, V1D(input[0][0].size() / fs)));
    for (unsigned int i = 0; i < input.size(); i++) {
        for (unsigned int j = 0; j < input[0].size(); j += fs) {
            for (unsigned int k = 0; k < input[0][0].size(); k += fs) {
                output[i][j / fs][k / fs] = std::max(std::max(input[i][j][k], input[i][j + 1][k]), std::max(input[i][j][k + 1], input[i][j + 1][k + 1]));
            }
        }
    }
    return output;
}

V3D MaxPool::calc_grad_z(V3D w_nxt, V3D grad_z_nxt) {
    V3D grad_z_cur(grad_z_nxt.size(), V2D(grad_z_nxt[0].size() * fs, V1D(grad_z_nxt[0][0].size() * fs)));
    for (unsigned int i = 0; i < grad_z_nxt.size(); i++) {
        for (unsigned int j = 0; j < grad_z_nxt[0].size(); j++) {
            for (unsigned int k = 0; k < grad_z_nxt[0][0].size(); k++) {
                for (int l = 0; l < fs; l++) {
                    for (int m = 0; m < fs; m++) {
                        grad_z_cur[i][j * fs + l][k * fs + m] = grad_z_nxt[i][j][k];
                    }
                }
            }
        }
    }
    return grad_z_cur;
}

void MaxPool::update_weights() {
    // No weights to update
}
