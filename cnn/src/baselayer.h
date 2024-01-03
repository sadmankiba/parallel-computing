#ifndef BASELAYER_H
#define BASELAYER_H

typedef std::vector<float> V1D;
typedef std::vector<V1D> V2D;
typedef std::vector<V2D> V3D;
typedef std::vector<V3D> V4D;

#define CONV_TYPE 'C'
#define POOL_TYPE 'P'
#define FLAT_TYPE 'F'
#define DENSE_TYPE 'D'

class BaseLayer {
protected:
    float sigmoid(float x);

public:
    float lr;
    char type;
    bool use_cpu;
    std::vector<int> size_cur;

    virtual void init_weights() = 0;
    void update_weights();

    float activ_forward(float z);
    float activ_grad(float z);
    float activ_grad_from_a(float a);

    void print_weights();
};

#endif