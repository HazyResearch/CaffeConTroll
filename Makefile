include .config
UNAME := $(shell uname)

LIBS=lmdb glog $(BLAS_LIBS) 
LD_BASE=$(foreach l, $(LIBS), -l$l)

INCLUDE_DIRS=$(BOOST_INCLUDE) $(GTEST_INCLUDE) $(GLOG_INCLUDE) $(GFLAGS_INCLUDE) \
	     $(LMDB_INCLUDE) $(BLAS_INCLUDE)
INCLUDE_STR=$(foreach d, $(INCLUDE_DIRS), -I$d)

LIB_DIRS=$(BOOST_LIB_DIR) $(GTEST_LIB_DIR) $(GLOG_LIB_DIR) $(GFLAGS_LIB_DIR) \
	 $(LMDB_LIB_DIR) $(BLAS_LIB_DIR)
LIB_STR=$(foreach d, $(LIB_DIRS), -L$d)

# For Mac OS X 10.10 x86_64 Yosemite
ifeq ($(UNAME), Darwin)
  CFLAGS = -Wall -std=c++11
  LDFLAGS = $(LD_BASE) -lboost_program_options-mt -lboost_serialization
# For Ubuntu 12.04 x86_64 (raiders3 machine)
else ifeq ($(UNAME), Linux)
  CFLAGS = -Wall -std=c++11 -Wl,--no-as-needed
  LDFLAGS = $(LD_BASE) -lrt -lboost_program_options -lboost_serialization
endif
CFLAGS += $(BLAS_DEFS)

DEBUG_FLAGS = -g -O0 -DDEBUG
ifeq ($(UNAME), Darwin)
  DEBUG_FLAGS += -ferror-limit=10
endif
ASSEMBLY_FLAGS= -S

DIR_PARAMS=$(INCLUDE_STR) $(LIB_STR)
PROTOBUF     = `pkg-config --cflags protobuf`
PROTOBUF_LIB = `pkg-config --cflags --libs protobuf`
WARNING_FLAGS = -Wextra
PRODUCT_FLAGS = -Ofast

# Protobuf variables
PROTO_SRC_DIR=src/parser/
PROTO_CC=protoc --cpp_out=.
PROTO_SRC=cnn.proto
PROTO_COMPILED_SRC=$(PROTO_SRC_DIR)cnn.pb.cc

# SOURCE FILE FOR MAIN PROGRAM
TARGET = caffe-ct
SRC = src/main.cpp src/DeepNet.cpp src/bridges/PhysicalStratum_impl.cpp \
      src/parser/parser.cpp src/parser/corpus.cpp src/util.cpp src/timer.cpp 
OBJ_FILES = $(patsubst %.cpp,%.o,$(SRC))

# SOURCE FILE FOR TEST
TEST_LDFLAGS= $(LDFLAGS) -L$(GTEST_LIB_DIR) -lgtest -lpthread 
TEST_SOURCES = src/DeepNet.cpp src/bridges/PhysicalStratum_impl.cpp \
	       src/parser/parser.cpp src/parser/corpus.cpp src/util.cpp src/timer.cpp tests/test_main.cpp \
	       tests/test_lrn_bridge.cpp tests/test_ReLU_bridge.cpp tests/test_MaxPooling_bridge.cpp \
	       tests/test_connector.cpp tests/test_model_write.cpp \
	       tests/test_softmax_bridge.cpp tests/test_dropout_bridge.cpp tests/test_cube.cpp \
	       tests/test_report.cpp tests/test_kernel.cpp tests/test_scanner.cpp \
	       tests/test_fc_bridge.cpp tests/test_grouping.cpp \
	       tests/test_parallelized_convolution.cpp tests/test_dropout_bridge.cpp tests/test_model_write.cpp \
	       tests/test_lenet_network.cpp
	       #snapshot-parser/simple_parse.cpp tests/test_imagenet_snapshot.cpp \

TEST_OBJ_FILES = $(patsubst %.cpp,%.o,$(TEST_SOURCES))
TEST_EXECUTABLE=test

.PHONY: all assembly clean product test warning

all: CFLAGS += -O0 -g 
all: $(OBJ_FILES) cnn.pb.o
	$(CC) $^ -o $(TARGET) $(DEBUG_FLAGS) $(CFLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

release: CFLAGS += $(PRODUCT_FLAGS)
release: $(OBJ_FILES) cnn.pb.o
	$(CC) $^ -o $(TARGET) $(CFLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

profile: CFLAGS += -D_DETAILED_PROFILING $(PRODUCT_FLAGS)
profile: $(OBJ_FILES) cnn.pb.o
	$(CC) $^ -o $(TARGET) $(CFLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

test: CFLAGS += -O0 -g -I $(GTEST_INCLUDE)
test: $(TEST_OBJ_FILES) cnn.pb.o 
	$(CC) $^ -o $(TEST_EXECUTABLE) $(DEBUG_FLAGS) $(CFLAGS) $(DIR_PARAMS) $(TEST_LDFLAGS) $(PROTOBUF_LIB) 

%.o: %.cpp $(PROTO_COMPILED_SRC)
	$(CC) $(CFLAGS) $(INCLUDE_STR) $(TEST_BLASFLAGS) $(PROTOBUF) -c $< -o $@

cnn.pb.o: $(PROTO_COMPILED_SRC)
	$(CC) $(CFLAGS) $(INCLUDE_STR) $(TEST_BLASFLAGS) $(PROTOBUF) -c $(PROTO_COMPILED_SRC)

$(PROTO_COMPILED_SRC): $(PROTO_SRC_DIR)$(PROTO_SRC)
	cd $(PROTO_SRC_DIR); $(PROTO_CC) $(PROTO_SRC); cd -

warning:
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(WARNING_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC) -o $(TARGET)

assembly:
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $(ASSEMBLY_FLAGS) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF) $(SRC)

clean:
	rm -f $(TARGET)
	rm -f $(PROTO_SRC_DIR)*.pb.*
	rm -f $(TEST_OBJ_FILES)
	rm -f $(OBJ_FILES)
	rm -f $(TEST_EXECUTABLE)
	rm -f tests/toprocess.bin tests/model.bin
