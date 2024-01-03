#include <cmath>

#include "mseloss.h"

float MSELoss::forward(float pred, float actual) {
    return pow(pred - actual, 2);
}


float MSELoss::grad(float pred, float actual) {
    return 2 * (pred - actual);
}

