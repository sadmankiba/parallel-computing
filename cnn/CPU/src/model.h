#ifndef MODEL_H
#define MODEL_H

#include "base.h"

class Model: public Base{
    public:
        Base** layers;
        int numLayers;
        
        Model(Base** _layers, int _numLayers);
        void forward(float *input, float *output, int numData);
        void update(int numData);
};

#endif