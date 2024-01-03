#ifndef FLATTEN_H
#define FLATTEN_H

class Flatten: public BaseLayer {
private:
    std::vector<int> in_size;

    void init_weights() override {}

public:
    Flatten();
    V1D forward(V3D input);

    V3D calc_grad_z(V1D grad_z_nxt);

    void update_weights();

    void set_in_size(std::vector<int> _in_size);

    void print_weights() {}
};

#endif 