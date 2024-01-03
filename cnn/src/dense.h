#ifndef DENSE_H
#define DENSE_H

class Dense: public BaseLayer {
private:
    int m; /* number of neurons in the previous layer */
    int n; /* number of neurons in the current layer */
    V1D input_last;
    V1D a_last;
    V1D grad_z_cur;

    void init_weights() override;

public:
    V2D weights;

    Dense(int m, int n);

    V1D forward(V1D input);

    V1D calc_grad_z(V2D w_nxt, V1D grad_z_nxt);

    void update_weights();  

    void set_grad_z(V1D grad_z);

    void print_weights() {}
};

#endif