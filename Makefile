CC = clang++
TARGET = deepnet
SRC = src/main.cpp src/parser/parser.cpp src/parser/cnn.pb.cc
DIRS=./lib/OpenBLAS/ ./lib/lmdb/
DIR_PARAMS=$(foreach d, $(DIRS), -I$d -L$d)
PROTOBUF = `pkg-config --cflags --libs protobuf`
LDFLAGS = -llmdb -lopenblas
CFLAGS = -Wall -std=c++11

ASSEMBLY_FLAGS= -S

DEBUG_FLAGS = -g -O0 -ferror-limit=10

WARNING_FLAGS = -Wextra

PRODUCT_FLAGS = -O3

TEST_CC=clang++
TEST_CFLAGS=-O2 -std=c++11

TEST_LDFLAGS=-I./lib/gtest-1.7.0/include/ -L./lib/gtest/ -lgtest -lpthread -L ./lib/OpenBLAS/ -lopenblas
TEST_BLASFLAGS=-lm -I ./lib/OpenBLAS/
TEST_SOURCES = tests/test_main.cpp tests/test_softmax_bridge.cpp 
#tests/test_convolution_bridge.cpp tests/test_MaxPooling_bridge.cpp tests/test_ReLU_bridge.cpp
TEST_EXECUTABLE=test

.PHONY: all assembly clean product test warning

all: 
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)

assembly:
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(ASSEMBLY_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC)

clean:
	rm $(TARGET)

product:
	$(CC) $(CFLAGS) $(PRODUCT_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)

test:
	$(TEST_CC) $(TEST_CFLAGS) $(TEST_SOURCES) $(TEST_LDFLAGS) $(TEST_BLASFLAGS) -o $(TEST_EXECUTABLE)

warning:
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(WARNING_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)
