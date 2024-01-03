#pragma once


enum LossType {
    LOGLOSS,
    MSELOSS
};

class CNNModel {
private:
    std::vector<BaseLayer *> layers;
    std::vector<int> in_size;
    unsigned int out_size;
    float lr;
    unsigned int batch_size;
    V2D output; /* dim 1 - batch, dim 2 - output layer of one data */
    LossType loss_type;

public:
    CNNModel(std::vector<int> _in_size, int _out_size, float _lr, int _batch_size, LossType _loss_type); 

    void add_layer(BaseLayer *layer);
    void add_layer(Conv *conv);
    void add_layer(Flatten *flatten);

    V2D forward(MDVector vec);
    void backward(V2D target);
    void update_weights();
    
    float loss_forward(float pred, float actual);
    float loss_grad(float pred, float actual);

    void train_test(ImageDataset dataset, int n_epochs, float loss_max);
};