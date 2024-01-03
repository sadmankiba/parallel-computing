# Compiler options

CC = g++
CFLAGS = -g -std=c++17

NVCC = nvcc
NVCC_FLAGS =  

# CUDA library paths
CUDA_ROOT_DIR=/usr/local/cuda

CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart

# Project files

TEST_DIR = .
SRC_DIR = ../src
UTILS_DIR = ../utils
  
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(UTILS_DIR)/*.cpp) 
CUDA_SRCS = $(wildcard $(SRC_DIR)/*.cu)
TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)

OBJS = $(CPP_SRCS:.cpp=.o)   
OBJS += $(CUDA_SRCS:.cu=.o)
OBJS += $(TEST_SRCS:.cpp=.o)

TARGET = test-cnn

# Make shortcuts
# $< = first dependency
# $@ = target
# $^ = all dependencies

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o : %.cu %.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 

clean:
	rm -f $(OBJS) $(TARGET)