UNAME := $(shell uname)

# For Mac OS X 10.10 x86_64 Yosemite
ifeq ($(UNAME), Darwin)
  CC = clang++
  CFLAGS = -Wall -std=c++11
  LDFLAGS = -llmdb -lopenblas
  DIRS=./externals/OpenBLAS/ ./lib/lmdb/
# For Ubuntu 12.04 x86_64 (raiders3 machine)
else ifeq ($(UNAME), Linux)
  CC = g++-4.8
  CFLAGS = -Wall -std=c++11 -Wl,--no-as-needed
  LDFLAGS = -llmdb -lopenblas -lrt
  DIRS=./externals/OpenBLAS/
endif
TARGET = deepnet
SRC = src/main.cpp src/parser/parser.cpp src/parser/cnn.pb.cc src/parser/corpus.cpp src/util.cpp
DIR_PARAMS=$(foreach d, $(DIRS), -I$d -L$d)
PROTOBUF = `pkg-config --cflags --libs protobuf`

ASSEMBLY_FLAGS= -S

DEBUG_FLAGS = -g -O0 -DDEBUG
ifeq ($(UNAME), Darwin)
  DEBUG_FLAGS += -ferror-limit=10
endif

WARNING_FLAGS = -Wextra

PRODUCT_FLAGS = -O3

TEST_CC=clang++
TEST_CFLAGS=-O2 -std=c++11

TEST_LDFLAGS= -lrt -I./lib/gtest-1.7.0/include/ -L./lib/gtest/ -lgtest -lpthread -L ./externals/OpenBLAS/ -lopenblas
TEST_BLASFLAGS= -lm -I ./externals/OpenBLAS/
TEST_SOURCES = tests/test_main.cpp tests/test_MaxPooling_bridge.cpp tests/test_softmax_bridge.cpp tests/test_convolution_bridge.cpp tests/test_ReLU_bridge.cpp tests/test_dropout_bridge.cpp tests/test_lrn_bridge.cpp src/util.cpp 
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
