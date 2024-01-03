#include <vector>
#include <cstdlib>

#include "baselayer.h"
#include "dense.h"

void Dense::init_weights() {
    weights.resize(m, V1D(n));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            weights[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

Dense::Dense(int m, int n) : m(m), n(n) {
    type = DENSE_TYPE;
    init_weights();
    grad_z_cur.resize(n);
}
V1D Dense::forward(V1D input) {
    input_last = input;
    V1D a(n);

    for (int i = 0; i < n; i++) {
        float z = 0;
        for (int j = 0; j < m; j++) {
            z += input[j] * weights[j][i];
        }
        a[i] = sigmoid(z);
    }
    a_last = a;
    return a;
}

V1D Dense::calc_grad_z(V2D w_nxt, V1D grad_z_nxt) {
    for (int i = 0; i < m; i++) {
        float grad_a = 0;
        for (int j = 0; j < n; j++) {
            grad_a += grad_z_nxt[j] * w_nxt[i][j];
        }
        grad_z_cur[i] = grad_a * a_last[i] * (1 - a_last[i]);
    }

    return grad_z_cur;
}

void Dense::update_weights() {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            weights[i][j] += lr * input_last[i] * grad_z_cur[j];
        }
    }
}  

void Dense::set_grad_z(V1D grad_z) {
    grad_z_cur = grad_z;
}