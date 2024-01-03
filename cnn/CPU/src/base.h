#ifndef BASE_H
#define BASE_H

class Base{
    public:
        float *input, *output;
        int numOut=-1; //number of output nodes

        virtual void forward(float *input, float *output, int numData) = 0;
        virtual void backward(int numData){};
        virtual void update(int numData){};
        void forward(float *input, float *output){};
        void backward(){};
        void update();
};

#endif