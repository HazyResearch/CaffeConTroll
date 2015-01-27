CC = clang++
TARGET = deepnet
SRC = src/main.cpp src/parser/parser.cpp src/parser/cnn.pb.cc
DIRS=./lib/OpenBLAS/ ./lib/lmdb/
DIR_PARAMS=$(foreach d, $(DIRS), -I$d -L$d)
PROTOBUF = `pkg-config --cflags --libs protobuf`
LDFLAGS = -llmdb -lopenblas
CFLAGS = -Wall -std=c++11
DEBUG_FLAGS = -g -O0 -ferror-limit=10

PRODUCT_FLAGS = -O3

TEST_CC=g++
TEST_CFLAGS=-O2 -std=c++0x
TEST_LDFLAGS=-lgtest -lpthread -lglog -lrt -L/home/shubham/Documents/research/OpenBLAS/ -lopenblas
TEST_BLASFLAGS=-lblas -lm -I/home/shubham/Documents/research/OpenBLAS/
TEST_SOURCES = tests/test_main.cpp tests/test_convolution_bridge.cpp tests/test_MaxPooling_bridge.cpp tests/test_ReLU_bridge.cpp
TEST_EXECUTABLE=test

.PHONY: all product clean test

all: 
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)

product:
	$(CC) $(CFLAGS) $(PRODUCT_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)

test:
	$(TEST_CC) $(TEST_CFLAGS) $(TEST_SOURCES) $(TEST_LDFLAGS) $(TEST_BLASFLAGS) -o $(TEST_EXECUTABLE)

clean:
	rm $(TARGET)
