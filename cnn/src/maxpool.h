#ifndef MAXPOOL_H
#define MAXPOOL_H

class MaxPool: public BaseLayer {
private:
    int fs; /* size of the filter, assuming square */

    void init_weights() override {}
public:
    MaxPool(int fs);
    V3D forward(V3D input);

    V3D calc_grad_z(V3D w_nxt, V3D grad_z_nxt);

    void update_weights();

    void print_weights() {}
};

#endif