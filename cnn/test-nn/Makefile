CC = g++
CFLAGS = -g -std=c++17 -Wall

MAIN_DIR = .
SRC_DIR = ../src
UTILS_DIR = ../utils

MAIN_SRCS = $(wildcard $(MAIN_DIR)/*.cpp) 
OTHER_SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(UTILS_DIR)/*.cpp) 

OBJS = $(MAIN_SRCS:.cpp=.o) $(OTHER_SRCS:.cpp=.o)

TARGET = test.out

all: clean $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)