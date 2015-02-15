include .config
UNAME := $(shell uname)

LIBS=lmdb openblas glog 
LD_BASE=$(foreach l, $(LIBS), -l$l)

INCLUDE_DIRS=$(BOOST_INCLUDE) $(GTEST_INCLUDE) $(LMBD_INCLUDE) $(OPENBLAS_INCLUDE)  
INCLUDE_STR=$(foreach d, $(INCLUDE_DIRS), -I$d)

LIB_DIRS=$(BOOST_LIB_DIR) $(GTEST_LIB_DIR) $(LMDB_LIBDIR) $(OPENBLAS_LIB_DIR)  
LIB_STR=$(foreach d, $(LIB_DIRS), -L$d)

# For Mac OS X 10.10 x86_64 Yosemite
ifeq ($(UNAME), Darwin)
  CFLAGS = -Wall -std=c++11
  LDFLAGS = $(LD_BASE) -lboost_program_options-mt
# For Ubuntu 12.04 x86_64 (raiders3 machine)
else ifeq ($(UNAME), Linux)
  CFLAGS = -Wall -std=c++11 -Wl,--no-as-needed
  LDFLAGS = $(LD_BASE) -lrt -lboost_program_options
endif

# Protobuf variables
PROTO_SRC_DIR=src/parser/
PROTO_CC=protoc --cpp_out=.
PROTO_SRC=cnn.proto
PROTO_COMPILED_SRC=$(PROTO_SRC_DIR)cnn.pb.cc

TARGET = deepnet
SRC = src/main.cpp src/bridges/PhysicalStratum_impl.cpp src/parser/parser.cpp src/parser/corpus.cpp src/util.cpp src/timer.cpp $(PROTO_COMPILED_SRC)
DIR_PARAMS=$(INCLUDE_STR) $(LIB_STR)
PROTOBUF = `pkg-config --cflags --libs protobuf`

ASSEMBLY_FLAGS= -S

DEBUG_FLAGS = -g -O0 -DDEBUG
ifeq ($(UNAME), Darwin)
  DEBUG_FLAGS += -ferror-limit=10
endif

WARNING_FLAGS = -Wextra

PRODUCT_FLAGS = -O3

TEST_CFLAGS=-O0 -std=c++11 -I $(GTEST_INCLUDE)
TEST_LDFLAGS= $(LDFLAGS) -L$(GTEST_LIB_DIR) -lgtest -lpthread 

TEST_BLASFLAGS= -lm -I $(OPENBLAS_INCLUDE)
TEST_SOURCES = src/bridges/PhysicalStratum_impl.cpp src/parser/parser.cpp src/parser/corpus.cpp src/util.cpp src/timer.cpp tests/test_main.cpp\
	tests/test_lenet_network.cpp\
	# tests/test_accuracy.cpp\
	# tests/test_lenet_network.cpp\
	# tests/test_lrn_bridge.cpp\
	# tests/test_ReLU_bridge.cpp\
	# tests/test_connector.cpp\
	# tests/test_model_write.cpp\
	# tests/test_convolution_bridge.cpp\
	# tests/test_softmax_bridge.cpp\
	# tests/test_dropout_bridge.cpp\
	# tests/test_cube.cpp\
	# #TODO: if any of these other tests are included
	# a linker error occurs
	#tests/test_lenet_network.cpp

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
	$(CC) $(TEST_CFLAGS) $(TEST_SOURCES) $(PROTO_COMPILED_SRC) $(DIR_PARAMS) $(TEST_LDFLAGS) $(TEST_BLASFLAGS) $(PROTOBUF) -o $(TEST_EXECUTABLE)

warning:
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(WARNING_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)

$(PROTO_COMPILED_SRC): $(PROTO_SRC_DIR)$(PROTO_SRC)
	cd $(PROTO_SRC_DIR); $(PROTO_CC) $(PROTO_SRC); cd -
