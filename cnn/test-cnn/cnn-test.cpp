#include <vector>
#include <cmath>
#include <iostream>
#include <cstring>
#include <cstdlib> 

#include "../src/baselayer.h"
#include "../src/mdvector.h"
#include "../src/imdata.h"
#include "../src/mnist.h"
#include "../src/conv.h"
#include "../src/flatten.h"
#include "../src/dense.h"
#include "../src/cnnmodel.h"
#include "loss-test.h"

#define DEBUG 0

void test_conv(void) {
    V2D img = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    V2D filter = {{1, 2}, {3, 4}};
    V2D filter2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    Conv conv = Conv(1, 1, 2, true);
    V2D res = conv.conv(img, filter);

    if (res[0][0] != 37 || res[0][1] != 47 || res[1][0] != 67 || res[1][1] != 77) {
        std::cout << "Convolution CPU test 1 failed" << std::endl;
    }
    else {
        std::cout << "Convolution CPU test 1 passed" << std::endl;
    }

    conv.use_cpu = false;
    res = conv.conv(img, filter);

    if (res[0][0] != 37 || res[0][1] != 47 || res[1][0] != 67 || res[1][1] != 77) {
        std::cout << "Convolution GPU test 1 failed" << std::endl;
    }
    else {
        std::cout << "Convolution GPU test 1 passed" << std::endl;
    }

    conv.use_cpu = true;
    res = conv.conv(img, filter2);

    if (res[0][0] != 285) {
        std::cout << "Convolution CPU test 2 failed" << std::endl;
    }
    else {
        std::cout << "Convolution CPU test 2 passed" << std::endl;
    }

    conv.use_cpu = false;
    res = conv.conv(img, filter2);

    if (res[0][0] != 285) {
        std::cout << "Convolution GPU test 2 failed" << std::endl;
    }
    else {
        std::cout << "Convolution GPU test 2 passed" << std::endl;
    }
}

/* 
Test a small convolution and a flatten layer
*/
void test_conv_weight_update(bool use_cpu, unsigned int n_imgs = 1, float lr = 0.01, 
            int n_epochs = 50, int batch_size = 1, LossType loss_type = LossType::MSELOSS) {
    int n_channels = 1;
    int img_x = 3;
    int img_y = 3;
    int out_size = 1;
    int n_filter = 1;
    int filter_size = 3;
    float loss_max = 0.01;

    ImageDataset dataset(n_imgs, n_channels, img_x, img_y);
    dataset.generate();

    std::vector<int> in_size = {n_channels, img_x, img_y};
    CNNModel model(in_size, out_size, lr, batch_size, loss_type);
    Conv *conv = new Conv(n_channels, n_filter, filter_size, use_cpu);
    Flatten *flatten = new Flatten();
    
    model.add_layer(conv);
    model.add_layer(flatten);

    model.train_test(dataset, n_epochs, loss_max);
}

void test_conv_backprop(bool use_cpu, unsigned int n_imgs = 1, float lr = 1, int n_epochs = 200, 
        int batch_size = 1, LossType loss_type = LossType::MSELOSS) {
    int n_channels = 1;
    int img_x = 4;
    int img_y = 4;
    int out_size = 1;
    int n_filter1 = 1;
    int n_filter2 = 1;
    int filter1_size = 3;
    int filter2_size = 2;
    float loss_max = 0.01;

    ImageDataset dataset(n_imgs, n_channels, img_x, img_y);
    dataset.generate();

    std::vector<int> in_size = {n_channels, img_x, img_y};
    CNNModel model(in_size, out_size, lr, batch_size, loss_type);
    Conv *conv1 = new Conv(n_channels, n_filter1, filter1_size, true);
    Conv *conv2 = new Conv(n_channels, n_filter2, filter2_size, true);
    Flatten *flatten = new Flatten();
    
    model.add_layer(conv1);
    model.add_layer(conv2);
    model.add_layer(flatten);

    model.train_test(dataset, n_epochs, loss_max);
}

void test_conv_dense(int batch_size = 1, LossType loss_type = LossType::MSELOSS) {
    /* 
    Make a dataset of 3 images and labels.
    Each image has 2 channels and 3x3 pixels.
    */
    ImageDataset dataset(3, 2, 3, 3);
    dataset.generate();
    float lr = 0.01;

    std::vector<int> in_size = {2, 3, 3};
    CNNModel model(in_size, 1, lr, batch_size, loss_type);
    Flatten flatten = Flatten();
    Dense dense = Dense(2*3*3, 1);
    model.add_layer(&flatten);
    model.add_layer(&dense);

    model.train_test(dataset, 1000, 0.01);
}

void test_mnist_small(bool use_cpu, unsigned int n_imgs = 1, float lr = 1, int n_epochs = 200, 
        int batch_size = 1, LossType loss_type = LossType::MSELOSS) {
    MNIST mnist;
    unsigned int n_classes = 2;
    unsigned int n_class_imgs = n_imgs / n_classes;
    unsigned int class_label;
    unsigned int mnist_img_size = 28;

    /* Get N images from two classes */
    float *classImages = new float[n_class_imgs * n_classes * mnist_img_size * mnist_img_size];
    float *labels = new float[n_class_imgs * n_classes];

    class_label = 0;
    mnist.readClass("train", class_label, n_class_imgs, classImages);
    if (DEBUG) {
        for (int i = 0; i < 20; i++) {
            std::cout << "Image " << i << std::endl;
            mnist.printImage(classImages + i * mnist_img_size * mnist_img_size, mnist_img_size);
            std::cout << std::endl;
        }
    }
    for (int i = 0; i < n_class_imgs; ++i) {
        labels[i] = class_label;
    }

    class_label = 1;
    mnist.readClass("train", class_label, n_class_imgs, classImages + n_class_imgs * mnist_img_size * mnist_img_size);

    for (int i = n_class_imgs; i < n_class_imgs * n_classes; ++i) {
        labels[i] = class_label;
    }

    /* Reduce image sizes */
    unsigned int reduced_img_size = 7;
    float *reducedImages = new float[n_class_imgs * n_classes * reduced_img_size * reduced_img_size];
    for (int i = 0; i < n_class_imgs * n_classes; ++i) {
        mnist.reduceImage(classImages + i * mnist_img_size * mnist_img_size, reducedImages + i * reduced_img_size * reduced_img_size, mnist_img_size, reduced_img_size);
    }

    if (DEBUG) {
        for (int i = 0; i < 20; i++) {
            std::cout << "Reduced image " << i << std::endl;
            mnist.printImage(reducedImages + i * reduced_img_size * reduced_img_size, reduced_img_size);
            std::cout << std::endl;
        }
    }
    
    /* Train and test model */
    int n_channels = 1;
    int img_x = 7;
    int img_y = 7;
    int out_size = 1;
    int n_filter1 = 1;
    int n_filter2 = 1;
    int filter1_size = 4;
    int filter2_size = 4;
    float loss_max = loss_type == LossType::MSELOSS ? 0.2 : 1.0;

    ImageDataset dataset(n_class_imgs * n_classes, n_channels, img_x, img_y);
    dataset.build(reducedImages, labels, n_class_imgs * n_classes);

    if(DEBUG) {
        std::cout << "Dataset size: " << dataset.images.size() << std::endl;
    }

    std::vector<int> in_size = {n_channels, img_x, img_y};
    CNNModel model(in_size, out_size, lr, batch_size, loss_type);
    Conv *conv1 = new Conv(n_channels, n_filter1, filter1_size, use_cpu);
    Conv *conv2 = new Conv(n_channels, n_filter2, filter2_size, use_cpu);
    Flatten *flatten = new Flatten();
    
    model.add_layer(conv1);
    model.add_layer(conv2);
    model.add_layer(flatten);

    model.train_test(dataset, n_epochs, loss_max); // TO-DO: Pass train and test dataset
}

void test_mnist(bool use_cpu, unsigned int n_imgs = 1, float lr = 1, int n_epochs = 200, 
        int batch_size = 1, LossType loss_type = LossType::MSELOSS) {
    MNIST mnist;
    unsigned int n_classes = 2;
    unsigned int n_class_imgs = n_imgs / n_classes;
    unsigned int class_label;
    unsigned int mnist_img_size = 28;

    /* Get N images from two classes */
    float *classImages = new float[n_class_imgs * n_classes * mnist_img_size * mnist_img_size];
    float *labels = new float[n_class_imgs * n_classes];

    class_label = 0;
    mnist.readClass("train", class_label, n_class_imgs, classImages);
    if (DEBUG) {
        for (int i = 0; i < 20; i++) {
            std::cout << "Image " << i << std::endl;
            mnist.printImage(classImages + i * mnist_img_size * mnist_img_size, mnist_img_size);
            std::cout << std::endl;
        }
    }
    for (int i = 0; i < n_class_imgs; ++i) {
        labels[i] = class_label;
    }

    class_label = 1;
    mnist.readClass("train", class_label, n_class_imgs, classImages + n_class_imgs * mnist_img_size * mnist_img_size);

    for (int i = n_class_imgs; i < n_class_imgs * n_classes; ++i) {
        labels[i] = class_label;
    }
    
    /* Train and test model */
    int n_channels = 1;
    int img_x = 28;
    int img_y = 28;
    int n_filter1 = 1;
    int filter1_size = 5;
    int n_filter2 = 1;
    int filter2_size = 5;
    int n_filter3 = 1;
    int filter3_size = 5;
    int n_filter4 = 1;
    int filter4_size = 5;
    int n_filter5 = 1;
    int filter5_size = 5;
    int n_filter6 = 1;
    int filter6_size = 5;
    int n_filter7 = 1;
    int filter7_size = 4;
    int out_size = 1;
    float loss_max = loss_type == LossType::MSELOSS ? 0.2 : 1.0;

    ImageDataset dataset(n_class_imgs * n_classes, n_channels, img_x, img_y);
    dataset.build(classImages, labels, n_class_imgs * n_classes);

    std::vector<int> in_size = {n_channels, img_x, img_y};
    CNNModel model(in_size, out_size, lr, batch_size, loss_type);
    Conv *conv1 = new Conv(n_channels, n_filter1, filter1_size, use_cpu);
    Conv *conv2 = new Conv(n_channels, n_filter2, filter2_size, use_cpu);
    Conv *conv3 = new Conv(n_channels, n_filter3, filter3_size, use_cpu);
    Conv *conv4 = new Conv(n_channels, n_filter4, filter4_size, use_cpu);
    Conv *conv5 = new Conv(n_channels, n_filter5, filter5_size, use_cpu);
    Conv *conv6 = new Conv(n_channels, n_filter6, filter6_size, use_cpu);
    Conv *conv7 = new Conv(n_channels, n_filter7, filter7_size, use_cpu);
    
    Flatten *flatten = new Flatten();
    
    model.add_layer(conv1);
    model.add_layer(conv2);
    model.add_layer(conv3);
    model.add_layer(conv4);
    model.add_layer(conv5);
    model.add_layer(conv6);
    model.add_layer(conv7);
    model.add_layer(flatten);

    model.train_test(dataset, n_epochs, loss_max);
}

void test_cnn(int batch_size = 1, LossType loss_type = LossType::MSELOSS) {
    ImageDataset dataset(3, 2, 3, 3);
    dataset.generate();

    std::vector<int> in_size = {2, 3, 3};
    CNNModel model(in_size, 1, 0.01, batch_size, loss_type);
    Conv conv = Conv(2, 2, 2, true);
    Flatten flatten = Flatten();
    Dense dense = Dense(2*2*2, 1);
    model.add_layer(&conv);
    model.add_layer(&flatten);
    model.add_layer(&dense);

    model.train_test(dataset, 1000, 0.01);
}

int main(int argc, char** argv) {
    srand(time(NULL)); 

    /* default values */
    int use_cpu;
    unsigned int n_imgs = 0;
    unsigned int test_num = 1;
    float lr = 0.0f;
    int n_epochs = 0;
    int batch_size = 0;
    
    if (argc > 1) {
        use_cpu = strcmp(argv[1], "cpu") == 0 ? true : false;
    }
    if (argc > 2) {
        test_num = atoi(argv[2]);
    }
    if (argc > 3) {
        lr = atof(argv[3]);
    }
    if (argc > 4) {
        n_epochs = atoi(argv[4]);
    }
    if (argc > 5) {
        batch_size = atoi(argv[5]);
    } 
    if (argc > 6) {
        n_imgs = atoi(argv[6]);
    }

    /*
    Good Breakpoint
    set test_num, lr, n_epochs, batch_size
    */
    switch(test_num) {
        case 1:
            test_log_loss();
            break;
        case 2:
            /* 
            1 3x3 image, 1 3x3 filter, 1 output 
            #weights = 1*3*3 = 9
            */
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 50;
            batch_size = 1;
            test_conv_weight_update(use_cpu, 1, lr, n_epochs, batch_size, LossType::MSELOSS);
            break;
        case 3:
            /* 
            1 3x3 image, 1 3x3 filter, 1 output 
            #weights = 1*3*3 = 9
            */
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 150;
            batch_size = 1;
            test_conv_weight_update(use_cpu, 1, lr, n_epochs, batch_size, LossType::LOGLOSS);
            break;
        case 4:
            /* 
            4 3x3 images, 1 3x3 channel, 1 output 
            #weights = 1*3*3 = 9
            */
            n_imgs = 4;
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 2000;
            batch_size = 1;
            test_conv_weight_update(use_cpu, n_imgs, lr, n_epochs, batch_size, LossType::MSELOSS);
            break;
        case 5:
            /* 
            4 3x3 images, 1 3x3 channel, 1 output 
            #weights = 1*3*3 = 9
            */
            n_imgs = 4;
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 5000;
            batch_size = 1;
            test_conv_weight_update(use_cpu, n_imgs, lr, n_epochs, batch_size, LossType::LOGLOSS);
            break;
        case 6:
            /* 
            4 3x3 images, 1 3x3 channel, 1 output
            #weights = 1*3*3 = 9
            */
            n_imgs = 4;
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 5000;
            batch_size = 4;
            test_conv_weight_update(use_cpu, n_imgs, lr, n_epochs, batch_size, LossType::MSELOSS);
        case 7:
            /* 
            4 3x3 images, 1 3x3 channel, 1 output
            #weights = 1*3*3 = 9
            */
            n_imgs = 4;
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 5000;
            batch_size = 4;
            test_conv_weight_update(use_cpu, n_imgs, lr, n_epochs, batch_size, LossType::LOGLOSS);
            break;
        case 8:
            /*
            1 4x4 images, 1 3x3 channel, 1 2x2channel, 1 output
            #weights = 1*3*3 + 1*2*2 = 13
            */
            n_imgs = 1;
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 40;
            batch_size = 1;
            test_conv_backprop(use_cpu, n_imgs, lr, n_epochs, batch_size, LossType::MSELOSS);
            break;
        case 9:
            /*
            1 4x4 images, 1 3x3 channel, 1 2x2channel, 1 output
            #weights = 1*3*3 + 1*2*2 = 13
            */
            n_imgs = 1;
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 80;
            batch_size = 1;
            test_conv_backprop(use_cpu, n_imgs, lr, n_epochs, batch_size, LossType::LOGLOSS);
            break;
        case 10:
            test_cnn();
            break;
        case 11:
            test_conv();
            break;
        case 12:
            /*
            200 7x7 images, 1 4x4 channel, 1 4x4channel, 1 output
            #weights = 1*4*4 + 1*4*4 = 32
            */
            n_imgs = 200;
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 80;
            batch_size = 1;
            test_mnist(use_cpu, n_imgs, lr, n_epochs, batch_size, LossType::MSELOSS);
            break;
        case 13: 
            n_imgs = n_imgs > 0 ? n_imgs : 400;
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 100;
            batch_size = 1;
            test_mnist_small(use_cpu, n_imgs, lr, n_epochs, batch_size, LossType::LOGLOSS);
            break;
        case 14: 
            n_imgs = n_imgs > 0 ? n_imgs : 400;
            lr = lr > 0 ? lr : 1;
            n_epochs = n_epochs > 0 ? n_epochs : 100;
            batch_size = 1;
            test_mnist(use_cpu, n_imgs, lr, n_epochs, batch_size, LossType::LOGLOSS);
            break;
        default:
            std::cout << "Invalid test number" << std::endl;
            break;
    }
    return 0;
}

