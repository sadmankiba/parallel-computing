
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <chrono>

#include "baselayer.h"
#include "mdvector.h"
#include "imdata.h"
#include "conv.h"
#include "flatten.h"
#include "maxpool.h"
#include "dense.h"
#include "mseloss.h"
#include "logloss.h"
#include "cnnmodel.h"

#define DEBUG 1
#define DEBUG2 0
#define DEBUG3 0

CNNModel::CNNModel(std::vector<int> _in_size, int _out_size, float _lr, int _batch_size, LossType _loss_type) : 
    in_size(_in_size), out_size(_out_size), lr(_lr), batch_size(_batch_size), loss_type(_loss_type) {
        output.resize(batch_size, V1D(out_size));
    } 

void CNNModel::add_layer(BaseLayer *layer) {
    layer->lr = lr;
    layers.push_back(layer);
    std::cout << "Added layer " << layer->type << std::endl;
}

void CNNModel::add_layer(Conv *conv) {
    if (layers.size() == 0) {
        conv->set_in_size(in_size);
    } else {
        Conv *prev_layer = dynamic_cast<Conv *>(layers.back());
        conv->set_in_size(prev_layer->size_cur);
    }
    conv->lr = lr;
    layers.push_back(conv);
    std::cout << "Added layer " << conv->type << std::endl;
}

void CNNModel::add_layer(Flatten *flatten) {
    Conv *prev_layer = dynamic_cast<Conv *>(layers.back());
    flatten->set_in_size(prev_layer->size_cur);
    flatten->lr = lr;
    layers.push_back(flatten);
    std::cout << "Added layer " << flatten->type << std::endl;
}

/*
Calculate output of a batch
*/
V2D CNNModel::forward(MDVector vec) {
    for (unsigned int j = 0; j < batch_size; j++) {
        /* This is fine because vec.v4 and vec.v2 is not overwritten by layers */
        if (vec.v4.size() > 0)
            vec.v3 = vec.v4[j];
        else if (vec.v2.size() > 0)
            vec.v1 = vec.v2[j];
        for (unsigned int i = 0; i < layers.size(); i++) {
            if (layers[i]->type == CONV_TYPE) {
                Conv *layer = dynamic_cast<Conv *>(layers[i]);
                vec.v3 = layer->forward(vec.v3);
            } else if (layers[i]->type == FLAT_TYPE) {
                Flatten *layer = dynamic_cast<Flatten *>(layers[i]);
                vec.v1 = layer->forward(vec.v3);     
            } else if (layers[i]->type == POOL_TYPE) {
                MaxPool *layer = dynamic_cast<MaxPool *>(layers[i]);
                vec.v3 = layer->forward(vec.v3);
            } else { // DENSE_TYPE
                Dense *layer = dynamic_cast<Dense *>(layers[i]);
                vec.v1 = layer->forward(vec.v1); 
            }
        }
        output[j] = vec.v1;
    }
    // Assuming last layer is dense or flatten
    return output; 
}

/*
Update weights based on a batch
Input: target, dim 1 - batch size, dim 2 - output size
*/
void CNNModel::backward(V2D target) {
    assert(target.size() == batch_size);
    assert(output[0].size() == (target[0].size() == out_size));

    V1D grad_z_loss(out_size, 0.0);
    float activ_grad;

    MDVector w_vec;
    MDVector grad_z_vec;

    /* Calculate loss gradient of z of each output by summing for a batch */
    for (unsigned int i = 0; i < out_size; i++) {
        for (unsigned int iinb = 0; iinb < batch_size; iinb++) {
            BaseLayer *last_layer = layers[layers.size() - 1];
            switch(last_layer->type) {
                case DENSE_TYPE:
                    activ_grad = (dynamic_cast<Dense *>(last_layer))->activ_grad_from_a(output[iinb][i]);
                    break;
                case FLAT_TYPE:
                    activ_grad = (dynamic_cast<Flatten *>(last_layer))->activ_grad_from_a(output[iinb][i]);
                    break;
                default:
                    std::cout << "Error: Last layer should be dense or flatten" << std::endl;
                    exit(-1);
            }
            
            grad_z_loss[i] += loss_grad(output[iinb][i], target[iinb][i])
             * activ_grad * 1.0 / batch_size;
        }
    }

    grad_z_vec.v1 = grad_z_loss;
    
    /* 
    Good Breakpoint
    Check output (whole batch), target (whole batch), loss, grad_loss
    */
    for (unsigned int i = layers.size(); i-- > 0;) {
        if (layers[i]->type == DENSE_TYPE) {
            Dense *layer = dynamic_cast<Dense *>(layers[i]);
            if (i == layers.size() - 1) {
                layer->set_grad_z(grad_z_vec.v1);
            } else {
                grad_z_vec.v1 = layer->calc_grad_z(w_vec.v2, grad_z_vec.v1);
                w_vec.v2 = layer->weights;
            }
        } else if (layers[i]->type == FLAT_TYPE) {
            Flatten *layer = dynamic_cast<Flatten *>(layers[i]);
            grad_z_vec.v3 = layer->calc_grad_z(grad_z_vec.v1);     
        } else if (layers[i]->type == CONV_TYPE) {
            // Assuming convolution layer is not last
            // and followed by either a conv or flatten layer
            Conv *layer = dynamic_cast<Conv *>(layers[i]);
            
            if (layers[i+1]->type == FLAT_TYPE) {
                layer->set_grad_z(grad_z_vec.v3);
            } else {
                grad_z_vec.v3 = layer->calc_grad_z(w_vec.v3, grad_z_vec.v3);
            }
            w_vec.v3 = layer->weights;
        } else if (layers[i]->type == POOL_TYPE) {
            
        } else {
            std::cout << "Error: Unknown layer type" << std::endl;
            exit(-1);
        }
    }
}

/*
Update weights based on a batch
*/
void CNNModel::update_weights() {
    for (unsigned int i = 0; i < layers.size(); i++) {
        if (layers[i]->type == CONV_TYPE) {
            (dynamic_cast<Conv *>(layers[i]))->update_weights();
        } else if (layers[i]->type == POOL_TYPE) {
            (dynamic_cast<MaxPool *>(layers[i]))->update_weights();
        } else if (layers[i]->type == FLAT_TYPE) {
            (dynamic_cast<Flatten *>(layers[i]))->update_weights();
        } else if (layers[i]->type == DENSE_TYPE) {
            (dynamic_cast<Dense *>(layers[i]))->update_weights();
        } else {
            std::cout << "Error: Unknown layer type" << std::endl;
            exit(-1);
        } 
    }
}

float CNNModel::loss_forward(float pred, float actual) {
    switch(loss_type) {
        case MSELOSS:
            return MSELoss().forward(pred, actual);
        case LOGLOSS:
            return LogLoss().forward(pred, actual);
        default:
            std::cout << "Error: Unknown loss type" << std::endl;
            exit(-1);
    }
}

float CNNModel::loss_grad(float pred, float actual) {
    switch(loss_type) {
        case MSELOSS:
            return MSELoss().grad(pred, actual);
        case LOGLOSS:
            return LogLoss().grad(pred, actual);
        default:
            std::cout << "Error: Unknown loss type" << std::endl;
            exit(-1);
    }
}

/* 
Train the model using backpropagation and test it 
*/
void CNNModel::train_test(ImageDataset dataset, int n_epochs, float loss_max) {
    
    MDVector vec;
    V2D target(batch_size, V1D(out_size));
    V2D output(batch_size, V1D(out_size));
    std::cout << "Training..." << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();
    /*
    Good Breakpoint
    Check dataset.images, dataset.labels
    */
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        
        for (int db = 0; db < (dataset.images.size() / batch_size); db++) {
            /* 
            Randomly choosing a batch to train.

            Iterating through all batches sequentially fails to 
            converge to a good solution. 
            */
            unsigned int bn = (1.0 * rand() / RAND_MAX) * (dataset.images.size() / batch_size);
            
            /* Calculate output of batch */
            V4D v(dataset.images.begin() + bn * batch_size, 
                dataset.images.begin() + (bn + 1) * batch_size);
            vec.v4 = v;
            output = this->forward(vec);
            
            if (DEBUG3) {
                std::cout << "Forward output: ";
                for (unsigned int b = 0; b < batch_size; b++) {
                    for (unsigned int j = 0; j < out_size; j++) {
                        std::cout << output[b][j] << " ";
                    }
                    std::cout << "  ";
                }
                std::cout << std::endl;
            }
            
            /* Calculate gradients based on batch output */
            V1D labels(dataset.labels.begin() + bn * batch_size, 
                dataset.labels.begin() + (bn + 1) * batch_size);
            
            if (DEBUG3) {
                std::cout << "Target: ";
            }
            for (unsigned int iinb = 0; iinb < batch_size; iinb++) {
                target[iinb][0] = labels[iinb];
                if (DEBUG3) std::cout << target[iinb][0] << " ";
            }
            
            if(DEBUG3) std::cout << std::endl;
            
            this->backward(target);

            if (DEBUG3) std::cout << "Backward done" << std::endl;
            
            /* Update weights */
            this->update_weights();
            
            if (DEBUG3) std::cout << "Weights updated" << std::endl;
        }
    }
    if (DEBUG)
        std::cout << "Training done" << std::endl;

    auto train_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(train_end - train_start).count();

    std::cout << "Total training time: " << (duration * 1.0) / 1000 << " ms" << std::endl;
    long long total_conv_time = 0;
    for (unsigned int i = 0; i < layers.size(); i++) {
        if (layers[i]->type == CONV_TYPE) {
            total_conv_time += (dynamic_cast<Conv *>(layers[i]))->conv_time;
        }
    }
    std::cout << "Total convolution time: " << (total_conv_time * 1.0) / 1000 << " ms" << std::endl;
    
    for (unsigned int i = 0; i < layers.size(); i++) {
        layers[i]->print_weights();
    }

    /* Test model */
    float loss = 0;
    if (DEBUG)
        std::cout << "Testing..." << std::endl;
    
    // TODO: Use loss on training and accuracy on test
    float accuracy = 0; /* Accuracy for two-class classification */
    for (unsigned int i = 0; i < dataset.images.size(); i++) {
        V4D v(dataset.images.begin() + i, 
            dataset.images.begin() + i + 1);
        vec.v4 = v;
        V2D output = this->forward(vec);
        if (DEBUG2) {
            std::cout << "Input: " << i+1 << " ";
            std::cout << "Output: " << output[0][0] << " ";
            std::cout << "Target: " << dataset.labels[i] << std::endl;
        }
        if (fabs(output[0][0] - dataset.labels[i]) < 0.5) {
            accuracy += 1.0 / dataset.images.size();
        }

        loss += loss_forward(output[0][0], dataset.labels[i]);
    }
    loss = loss / dataset.images.size();

    
    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
    
    if (loss < loss_max) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
}

