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
  LDFLAGS = $(LD_BASE) -lboost_program_options-mt -lboost_serialization -lpthread
  NVCCFLAGS = -D_GPU_TARGET -std=c++11 $(LD_BASE) -lcublas -lcuda -lboost_program_options-mt -lboost_serialization -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50
# For Ubuntu 12.04 x86_64 (raiders3 machine)
else ifeq ($(UNAME), Linux)
  #CFLAGS = -Wall -std=c++11 -Wl,--no-as-needed
  CFLAGS = -std=c++11 -I/usr/local/cuda-6.5/targets/x86_64-linux/include/
  NVCCFLAGS = -D_GPU_TARGET -std=c++11 $(LD_BASE) -lcublas -lcuda -lboost_program_options -lboost_serialization -I/usr/local/cuda-6.5/targets/x86_64-linux/include/ -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50
  LDFLAGS = $(LD_BASE) -lrt -lboost_program_options -lboost_serialization -lpthread 
endif
CFLAGS += $(BLAS_DEFS)

DEBUG_FLAGS = -g -O0 -DDEBUG
ifeq ($(UNAME), Darwin)
  DEBUG_FLAGS += -ferror-limit=10
endif
ASSEMBLY_FLAGS= -S

DIR_PARAMS=$(INCLUDE_STR) $(LIB_STR)
#PROTOBUF     = `pkg-config --cflags protobuf`
#PROTOBUF_LIB = `pkg-config --cflags --libs protobuf`
PROTOBUF_LIB = -lprotobuf
WARNING_FLAGS = -Wextra
PRODUCT_FLAGS = -Ofast

# Protobuf variables
PROTO_SRC_DIR=src/parser/
PROTO_CC=protoc --cpp_out=.
PROTO_SRC=cnn.proto
PROTO_COMPILED_SRC=$(PROTO_SRC_DIR)cnn.pb.cc

# SOURCE FILE FOR MAIN PROGRAM
TARGET = caffe-ct
SRC = src/DeepNetConfig.cpp src/util.cpp src/timer.cpp src/main.cpp src/sched/DeviceDriver_CPU.cpp
OBJ_FILES = $(patsubst %.cpp,%.o,$(SRC))

MAIN_CUDA_SOURCES = src/sched/DeviceDriver_GPU.cu
MAIN_CUDA_OBJ_FILES = $(patsubst %.cu,%.o,$(MAIN_CUDA_SOURCES))

# SOURCE FILE FOR TEST
#TEST_LDFLAGS= $(LDFLAGS) -L$(GTEST_LIB_DIR) -lgtest -lpthread 
TEST_LDFLAGS= $(LDFLAGS) -L$(GTEST_LIB_DIR) -lgtest
TEST_SOURCES = tests/test_main.cpp src/util.cpp src/timer.cpp src/DeepNetConfig.cpp \
	       src/sched/DeviceDriver_CPU.cpp \
	       tests/test_scanner.cpp \
	       #tests/test_parallelized_convolution.cpp \
	       #tests/test_fc_bridge.cpp \
	       #tests/test_MaxPooling_bridge.cpp \
	       #tests/test_ReLU_bridge.cpp \
	       #tests/test_softmax_bridge.cpp \
	       #tests/test_dropout_bridge.cpp \
	       #tests/test_lrn_bridge.cpp \
	       #tests/test_cube.cpp \
	       #tests/test_report.cpp \
	       #tests/test_grouping.cpp \
	       tests/test_lenet_network.cpp

	       # tests/test_device_driver.cpp \
	       tests/test_device_driver_gpu.cpp \
	       tests/test_connector.cpp \
	       tests/test_model_write.cpp \
	       tests/test_kernel.cpp \
	       tests/test_scanner.cpp \
	       tests/test_convolution.cpp \
# snapshot-parser/simple_parse.cpp tests/test_imagenet_snapshot.cpp \

TEST_OBJ_FILES = $(patsubst %.cpp,%.o,$(TEST_SOURCES))
TEST_EXECUTABLE=test

#TEST_CUDA_SOURCES = src/sched/DeviceDriver_GPU.cu \
#					tests/test_convolution.cu
					
TEST_CUDA_OBJ_FILES = $(patsubst %.cu,%.o,$(TEST_CUDA_SOURCES))

SNAPSHOT_SOURCES = src/util.cpp src/timer.cpp tests/test_main.cpp \
                   snapshot-parser/simple_parse.cpp tests/test_imagenet_snapshot.cpp \

SNAPSHOT_OBJ_FILES = $(patsubst %.cpp,%.o,$(SNAPSHOT_SOURCES))
SNAPSHOT_EXECUTABLE=snapshot

ifdef NVCC
LINKCC = $(NVCC)
LINKFLAG = $(NVCCFLAGS)
else
LINKCC = $(CC)
LINKFLAG = $(CFLAGS) $(LDFLAGS)
endif

.PHONY: all assembly clean product test warning

all: CFLAGS += $(DEBUG_FLAGS) 
all: LINKFLAG += $(DEBUG_FLAGS) 
all: $(OBJ_FILES) cnn.pb.o $(MAIN_CUDA_OBJ_FILES)
	$(LINKCC) $^ -o $(TARGET) $(LINKFLAG) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

release: CFLAGS += $(PRODUCT_FLAGS)
release: LINKFLAG += $(PRODUCT_FLAGS)
release: $(OBJ_FILES) cnn.pb.o 
	$(LINKCC) $^ -o $(TARGET) $(LINKFLAG) $(BUILDFLAG) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

profile: CFLAGS += -D_DETAILED_PROFILING -D_FASTPOW  $(PRODUCT_FLAGS)
profile: LINKFLAG += -D_DETAILED_PROFILING -D_FASTPOW  $(PRODUCT_FLAGS)
profile: $(OBJ_FILES) cnn.pb.o
	$(LINKCC) $^ -o $(TARGET) $(LINKFLAG) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

test: CFLAGS += $(DEBUG_FLAGS) -I $(GTEST_INCLUDE)
test: LINKFLAG += $(DEBUG_FLAGS) -I $(GTEST_INCLUDE)
test: $(TEST_OBJ_FILES) $(TEST_CUDA_OBJ_FILES) $(TEST_OBJ_FILES) cnn.pb.o 
	$(LINKCC) $^ -o $(TEST_EXECUTABLE) $(LINKFLAG) $(DIR_PARAMS) $(TEST_LDFLAGS) $(PROTOBUF_LIB) 

snapshot: CFLAGS += $(PRODUCT_FLAGS) -I $(GTEST_INCLUDE)
snapshot: LINKFLAG += $(PRODUCT_FLAGS) -I $(GTEST_INCLUDE)
snapshot: $(SNAPSHOT_OBJ_FILES) cnn.pb.o
	$(CC) $^ -o $(SNAPSHOT_EXECUTABLE) $(LINKFLAG) $(DIR_PARAMS) $(TEST_LDFLAGS) $(PROTOBUF_LIB) 

%.o: %.cpp $(PROTO_COMPILED_SRC)
	$(CC) $(CFLAGS) $(INCLUDE_STR) $(TEST_BLASFLAGS) $(PROTOBUF) -c $< -o $@

%.o: %.cu $(PROTO_COMPILED_SRC)
	$(NVCC) -O3 $(BLAS_DEFS) $(NVCCFLAGS) $(INCLUDE_STR) $(TEST_BLASFLAGS) -dc $< -o $@

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
	rm -f $(SNAPSHOT_OBJ_FILES)
	rm -f $(TEST_CUDA_OBJ_FILES)
	rm -f $(MAIN_CUDA_OBJ_FILES)
	rm -f $(SNAPSHOT_EXECUTABLE)
	rm -f tests/toprocess.bin tests/model.bin
