# CNN-GPU

## Neural Network

CPU implementation is in `CPU/` directory and GPU implementation is in `GPU/` directory. CPU implementation can be built with `CPU/tests/Makefile`. GPU implementation can be built with `GPU-MLP/test/Makefile` directory.

## Convolutional Neural Network

CPU and GPU share common code. The convolution implementation in CUDA is in `FinalProject759/src/convgpu.cu` file.

Can be built as 
```sh
cd FinalProject759/test-cnn/
make 
```

Tests can be run as 
```sh
cd test-cnn/
./test-cnn
```

Tests have default parameters that are tuned to pass the tests. Parameters can be updated from command line. 
```sh
# syntax: ./test-cnn <cpu | gpu> <test-number> <learning-rate> <num-epochs> <batch-size> <num_images>

# Example
./test-cnn cpu 4 0.1 200 8  # run on CPU, test-number 4, learning rate 0.1 #epochs 200, batch-size 8
```

The slurm script to run cnn is also present in the `test-cnn` directory.

### Tests

```
2-5: Single layer
6-7: Batching
8-9: Multi layer
12-14: MNIST
```

