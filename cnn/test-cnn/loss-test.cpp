#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

#include "../src/baselayer.h"
#include "../src/mdvector.h"
#include "../src/imdata.h"
#include "../src/conv.h"
#include "../src/flatten.h"
#include "../src/dense.h"
#include "../src/cnnmodel.h"
#include "../src/logloss.h"

#define EPS 0.001

void test_log_loss(void) {
    LogLoss logloss;

    float actual = 0;
    float pred = 0.83;

    assert(abs(logloss.forward(pred, actual) - 1.772) < EPS);
    assert(abs(logloss.grad(pred, actual) - 5.882) < EPS);

    actual = 0;
    pred = 0.001;

    assert(abs(logloss.forward(pred, actual) - 0.001) < EPS);
    assert(abs(logloss.grad(pred, actual) - 1.001) < EPS);

    actual = 1;
    pred = 0.024;

    assert(abs(logloss.forward(pred, actual) - 3.729) < EPS);
    assert(abs(logloss.grad(pred, actual) - (-41.667)) < EPS);

    actual = 1;
    pred = 0.6;

    assert(abs(logloss.forward(pred, actual) - 0.511) < EPS);
    assert(abs(logloss.grad(pred, actual) - (-1.667)) < EPS);

    std::cout << "Log loss test passed!" << std::endl;
}
