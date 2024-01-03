#pragma once

class MSELoss {
public:
    float forward(float pred, float actual);
    float grad(float pred, float actual);
};