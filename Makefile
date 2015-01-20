CC = clang++
TARGET = deepnet
SRC = src/main.cpp src/parser/parser.cpp src/parser/cnn.pb.cc
DIRS=./lib/OpenBLAS/ ./lib/lmdb/
DIR_PARAMS=$(foreach d, $(DIRS), -I$d -L$d)
PROTOBUF = `pkg-config --cflags --libs protobuf`
LDFLAGS = -llmdb -lopenblas
CFLAGS = -Wall -std=c++11
DEBUG_FLAGS = -g -O0
PRODUCT_FLAGS = -O3

.PHONY: all product clean

all: 
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)

product:
	$(CC) $(CFLAGS) $(PRODUCT_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)
clean:
	rm $(TARGET)
