#pragma once

class LogLoss {
public:
    float forward(float pred, float actual);
    float grad(float pred, float actual);
};