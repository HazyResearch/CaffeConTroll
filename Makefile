include .config
UNAME := $(shell uname)
DIRS=$(OPENBLAS_DIR) $(LMDB_DIR) 

# For Mac OS X 10.10 x86_64 Yosemite
ifeq ($(UNAME), Darwin)
  CFLAGS = -Wall -std=c++11
  LDFLAGS = -llmdb -lopenblas
# For Ubuntu 12.04 x86_64 (raiders3 machine)
else ifeq ($(UNAME), Linux)
  CFLAGS = -Wall -std=c++11 -Wl,--no-as-needed
  LDFLAGS = -llmdb -lopenblas -lrt -lglog -lboost_program_options
endif

# Protobuf variables
PROTO_SRC_DIR=src/parser/
PROTO_CC=protoc --cpp_out=.
PROTO_SRC=cnn.proto
PROTO_COMPILED_SRC=$(PROTO_SRC_DIR)cnn.pb.cc

TARGET = deepnet
SRC = src/main.cpp src/parser/parser.cpp src/parser/corpus.cpp src/util.cpp src/timer.cpp $(PROTO_COMPILED_SRC)
DIR_PARAMS=$(foreach d, $(DIRS), -I$d -L$d)
PROTOBUF = `pkg-config --cflags --libs protobuf`

ASSEMBLY_FLAGS= -S

DEBUG_FLAGS = -g -O0 -DDEBUG
ifeq ($(UNAME), Darwin)
  DEBUG_FLAGS += -ferror-limit=10
endif

WARNING_FLAGS = -Wextra

PRODUCT_FLAGS = -O3

TEST_CC= clang++
TEST_CFLAGS=-O0 -std=c++11 -I $(GTEST_INCLUDE)
TEST_LDFLAGS= $(LDFLAGS)   -L$(GTEST_LIB_DIR) -lgtest -lpthread -L $(OPENBLAS_DIR) -lopenblas

TEST_BLASFLAGS= -lm -I $(OPENBLAS_DIR)
TEST_SOURCES = src/util.cpp src/timer.cpp tests/test_main.cpp\
	tests/test_model_write.cpp\
	tests/test_convolution_bridge.cpp\
	tests/test_softmax_bridge.cpp\
	tests/test_dropout_bridge.cpp\
	# TODO: if any of these other tests are included
	# a linker error occurs
	#tests/test_lrn_bridge.cpp\
	#tests/test_MaxPooling_bridge.cpp\
	#tests/test_ReLU_bridge.cpp\
	#tests/test_parallelized_convolution.cpp
TEST_EXECUTABLE=test

.PHONY: all assembly clean product test warning

all: $(PROTO_COMPILED_SRC)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)

assembly:
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(ASSEMBLY_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC)

clean:
	rm -f $(TARGET)
	rm -f $(PROTO_SRC_DIR)*.pb.*

product:
	$(CC) $(CFLAGS) $(PRODUCT_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)

test: $(PROTO_COMPILED_SRC)
	$(TEST_CC) $(TEST_CFLAGS) $(TEST_SOURCES) $(PROTO_COMPILED_SRC) $(DIR_PARAMS) $(TEST_LDFLAGS) $(TEST_BLASFLAGS) $(PROTOBUF) -o $(TEST_EXECUTABLE)

warning:
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(WARNING_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)

$(PROTO_COMPILED_SRC): $(PROTO_SRC_DIR)$(PROTO_SRC)
	cd $(PROTO_SRC_DIR); $(PROTO_CC) $(PROTO_SRC); cd -
