NVCC = nvcc
NVCC_FLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17

MAIN_DIR = .
SRC_DIR = ../src

MAIN_SRCS = $(wildcard $(MAIN_DIR)/*.cu) $(wildcard $(MAIN_DIR)/*.cuh)
OTHER_SRCS = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.h)

OBJS_MODEL = $(filter-out $(MAIN_DIR)/unitTest.o, $(MAIN_SRCS:.cu=.o)) $(OTHER_SRCS:.cu=.o)
OBJS_UNIT = $(filter-out $(MAIN_DIR)/modelTest.o, $(MAIN_SRCS:.cu=.o)) $(OTHER_SRCS:.cu=.o)

TARGET_MODEL = nn

all: $(TARGET_MODEL)

$(TARGET_MODEL): $(OBJS_MODEL)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(MAIN_DIR)/*.o $(SRC_DIR)/*.o $(TARGET_MODEL)