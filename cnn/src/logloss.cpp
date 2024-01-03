#include <cmath>

#include "logloss.h"

float LogLoss::forward(float pred, float actual) {
    return -actual * log(pred) - (1 - actual) * log(1 - pred);
}

float LogLoss::grad(float pred, float actual) {
    return (pred - actual) / (pred * (1 - pred));
}