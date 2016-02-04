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
  CFLAGS = -Wall -std=c++11 -fsanitize-undefined-trap-on-error -fsanitize=integer-divide-by-zero
  DEBUG_FLAGS = -g -O0 -DDEBUG -ferror-limit=10
  LDFLAGS = $(LD_BASE) -lboost_program_options-mt -lboost_serialization -lpthread
  NVCC_DEBUG_FLAGS = -DDEBUG
  NVCCFLAGS = -D_GPU_TARGET -D_INCLUDE_GPUDRIVER -std=c++11 $(LD_BASE) -lcublas -lcuda -lboost_program_options-mt -lboost_serialization -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 -I $(CUDA_INCLUDE) -L $(CUDA_LIB)
# For Ubuntu 12.04 x86_64
else ifeq ($(UNAME), Linux)
  CFLAGS = -Wall -Wl,--no-as-needed -std=c++11
  DEBUG_FLAGS = -gdwarf-3 -O0 -DDEBUG # -gdwarf-3 necessary for debugging with gdb v7.4
  NVCC_DEBUG_FLAGS = -DDEBUG
  NVCCFLAGS = -D_GPU_TARGET -D_INCLUDE_GPUDRIVER -std=c++11 $(LD_BASE) -lcublas -lcuda -lboost_program_options -lboost_serialization -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 -I $(CUDA_INCLUDE) -L $(CUDA_LIB)
  LDFLAGS = $(LD_BASE) -lrt -lboost_program_options -lboost_serialization -lpthread 
endif
CFLAGS += $(BLAS_DEFS)

ASSEMBLY_FLAGS= -march=native -masm=intel -fverbose-asm -S

DIR_PARAMS=$(INCLUDE_STR) $(LIB_STR)
PROTOBUF_LIB = -lprotobuf
WARNING_FLAGS = -Wextra
PRODUCT_FLAGS = -Ofast -D_FASTPOW

# Protobuf variables
PROTO_SRC_DIR=src/parser/
PROTO_CC=protoc --cpp_out=.
PROTO_SRC=cnn.proto
PROTO_COMPILED_SRC=$(PROTO_SRC_DIR)cnn.pb.cc

# SOURCE FILE FOR MAIN PROGRAM
TARGET = caffe-ct
SRC = src/DeepNetConfig.cpp src/util.cpp src/timer.cpp src/main.cpp src/sched/DeviceDriver_CPU.cpp
OBJ_FILES = $(patsubst %.cpp,%.o,$(SRC))

ifdef NVCC
MAIN_CUDA_SOURCES = src/sched/DeviceDriver_GPU.cu
MAIN_CUDA_OBJ_FILES = $(patsubst %.cu,%.o,$(MAIN_CUDA_SOURCES))
CFLAGS += -D_INCLUDE_GPUDRIVER  -I $(CUDA_INCLUDE) -L $(CUDA_LIB)
endif

# SOURCE FILE FOR TEST
TEST_LDFLAGS= $(LDFLAGS) -L$(GTEST_LIB_DIR) -lgtest
TEST_SOURCES = tests/test_main.cpp src/util.cpp src/timer.cpp src/DeepNetConfig.cpp \
			src/sched/DeviceDriver_CPU.cpp \
			tests/test_MaxPooling_bridge.cpp \
			tests/test_parallelized_convolution.cpp \
			tests/test_lrn_bridge.cpp \
			tests/test_ReLU_bridge.cpp \
			tests/test_parallelized_fc.cpp \
			tests/test_parallelized_convolution_large_CPU.cpp \
			tests/test_parallelized_convolution_large_CPU_batch.cpp \
			tests/test_convolution.cpp \
			tests/test_fc_bridge.cpp \
			tests/test_kernel.cpp \
			tests/test_connector.cpp \
			tests/test_dropout_bridge.cpp \
			tests/test_softmax_bridge.cpp \
			tests/test_device_driver_cpu.cpp \
			tests/test_cube.cpp \
			tests/test_report.cpp \
			tests/test_grouping.cpp \
			tests/test_model_write.cpp \
			tests/test_scanner.cpp \
			tests/test_lenet_network.cpp \
			#tests/test_ReLU_bridge_GPU.cpp \
			tests/test_lrn_bridge_GPU.cpp \
			tests/test_MaxPooling_bridge_GPU.cpp \
			tests/test_parallelized_fc_GPU.cpp \
			tests/test_parallelized_convolution_GPU.cpp \
			tests/test_parallelized_convolution_large_GPU.cpp \
			tests/test_parallelized_convolution_large_GPU_batch.cpp \
			tests/test_parallelized_convolution_large_CPU_GPU_batch.cpp \
			tests/test_lenet_network_GPU.cpp \
			#tests/test_parallelized_convolution_2GPU.cpp \
			#tests/test_parallelized_convolution_large_2GPU.cpp \
			#tests/test_parallelized_convolution_large_2GPU_batch.cpp \
			#tests/test_parallelized_convolution_4GPU.cpp \
			#tests/test_lenet_network_4GPU.cpp \
			#tests/test_alexnet_1GPU.cpp \
			#tests/test_alexnet_1GPU_CPU.cpp \
			#tests/test_alexnet_4GPU.cpp \
			#tests/test_paper3a_conv_layer_GPU.cpp \
			#tests/test_paper3a_conv_layer_CPU_GPU.cpp \
			#tests/test_paper3a_conv_layer.cpp \
			#tests/test_paper3a_conv_layer_2GPU.cpp \
			#tests/test_paper3a_conv_layer_4GPU.cpp \
			#tests/test_paper3a_conv_layer_4GPU_CPU.cpp \
			#tests/test_paper3b_caffenet.cpp \
			#tests/test_paper3b_caffenet_1GPU.cpp \
			#tests/test_perf_convolution_1.cpp \
			#tests/test_perf_convolution_2.cpp \
			#tests/test_perf_convolution_3.cpp \
			#tests/test_perf_convolution_4.cpp \
			#tests/test_perf_convolution_5.cpp \
			#tests/test_perf_convolution_6.cpp \
			#tests/test_perf_convolution_7.cpp \
			#tests/test_perf_convolution_1_GPU.cpp \
			#tests/test_perf_convolution_2_GPU.cpp \
			#tests/test_perf_convolution_3_GPU.cpp \
			#tests/test_perf_convolution_4_GPU.cpp \
			#tests/test_perf_convolution_5_GPU.cpp \
			#tests/test_perf_convolution_6_GPU.cpp \
			#tests/test_perf_convolution_7_GPU.cpp \
			#tests/test_alexnet_network.cpp \
			#tests/test_device_driver_gpu.cpp \

TEST_OBJ_FILES = $(patsubst %.cpp,%.o,$(TEST_SOURCES))
TEST_EXECUTABLE=test

ifdef NVCC
TEST_CUDA_SOURCES = src/sched/DeviceDriver_GPU.cu \
					#tests/test_convolution.cu					
TEST_CUDA_OBJ_FILES = $(patsubst %.cu,%.o,$(TEST_CUDA_SOURCES))
endif

SNAPSHOT_SOURCES = tests/test_main.cpp src/util.cpp src/timer.cpp src/DeepNetConfig.cpp \
	       src/sched/DeviceDriver_CPU.cpp \
	       snapshot-parser/simple_parse.cpp tests/test_imagenet_snapshot.cpp \

SNAPSHOT_OBJ_FILES = $(patsubst %.cpp,%.o,$(SNAPSHOT_SOURCES))
SNAPSHOT_EXECUTABLE=snapshot

LINKCC = $(CC)
LINKFLAG = $(CFLAGS) $(LDFLAGS)

ifdef NVCC
LINKFLAG += -lcublas -lcudart -lcurand
NVCC_LINK = dlink.o
endif

.PHONY: all assembly clean product test warning

all: CFLAGS += $(PRODUCT_FLAGS) 
all: LINKFLAG += $(PRODUCT_FLAGS) 
all: $(OBJ_FILES) cnn.pb.o $(MAIN_CUDA_OBJ_FILES)
ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TARGET) $(LINKFLAG) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

assert: CFLAGS += $(PRODUCT_FLAGS) -D_DO_ASSERT
assert: LINKFLAG += $(PRODUCT_FLAGS) -D_DO_ASSERT
assert: $(OBJ_FILES) cnn.pb.o $(MAIN_CUDA_OBJ_FILES)

ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TARGET) $(LINKFLAG) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

release: CFLAGS += $(PRODUCT_FLAGS)
release: LINKFLAG += $(PRODUCT_FLAGS)
release: $(OBJ_FILES) cnn.pb.o $(MAIN_CUDA_OBJ_FILES)
ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TARGET) $(LINKFLAG) $(BUILDFLAG) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

profile: CFLAGS += -D_LAYER_PROFILING -D_FASTPOW  $(PRODUCT_FLAGS)
profile: LINKFLAG += -D_LAYER_PROFILING -D_FASTPOW  $(PRODUCT_FLAGS)
profile: $(OBJ_FILES) cnn.pb.o $(MAIN_CUDA_OBJ_FILES)
ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TARGET) $(LINKFLAG) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

profile2: CFLAGS += -D_DETAILED_PROFILING -D_LAYER_PROFILING -D_FASTPOW  $(PRODUCT_FLAGS)
profile2: LINKFLAG += -D_DETAILED_PROFILING -D_LAYER_PROFILING -D_FASTPOW  $(PRODUCT_FLAGS)
profile2: $(OBJ_FILES) cnn.pb.o $(MAIN_CUDA_OBJ_FILES)
ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TARGET) $(LINKFLAG) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)

test: CFLAGS += $(PRODUCT_FLAGS) -I $(GTEST_INCLUDE)
test: LINKFLAG += $(PRODUCT_FLAGS) -I $(GTEST_INCLUDE)
test: $(TEST_OBJ_FILES) $(TEST_CUDA_OBJ_FILES) $(TEST_OBJ_FILES) cnn.pb.o
ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TEST_EXECUTABLE) $(LINKFLAG) $(DIR_PARAMS) $(TEST_LDFLAGS) $(PROTOBUF_LIB)

test_debug: CFLAGS += $(DEBUG_FLAGS) -I $(GTEST_INCLUDE)
test_debug: NVCCFLAGS += $(NVCC_DEBUG_FLAGS)
test_debug: LINKFLAG += $(DEBUG_FLAGS) -I $(GTEST_INCLUDE)
test_debug: $(TEST_OBJ_FILES) $(TEST_CUDA_OBJ_FILES) $(TEST_OBJ_FILES) cnn.pb.o
ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TEST_EXECUTABLE) $(LINKFLAG) $(DIR_PARAMS) $(TEST_LDFLAGS) $(PROTOBUF_LIB)

test_profile: CFLAGS += -D_LAYER_PROFILING $(PRODUCT_FLAGS) -I $(GTEST_INCLUDE)
test_profile: LINKFLAG += -D_LAYER_PROFILING $(PRODUCT_FLAGS) -I $(GTEST_INCLUDE)
test_profile: $(TEST_OBJ_FILES) $(TEST_CUDA_OBJ_FILES) $(TEST_OBJ_FILES) cnn.pb.o
ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TEST_EXECUTABLE) $(LINKFLAG) $(DIR_PARAMS) $(TEST_LDFLAGS) $(PROTOBUF_LIB)

test_profile2: CFLAGS += -D_DETAILED_PROFILING -D_LAYER_PROFILING $(PRODUCT_FLAGS) -I $(GTEST_INCLUDE)
test_profile2: LINKFLAG += -D_DETAILED_PROFILING -D_LAYER_PROFILING $(PRODUCT_FLAGS) -I $(GTEST_INCLUDE)
test_profile2: $(TEST_OBJ_FILES) $(TEST_CUDA_OBJ_FILES) $(TEST_OBJ_FILES) cnn.pb.o
ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TEST_EXECUTABLE) $(LINKFLAG) $(DIR_PARAMS) $(TEST_LDFLAGS) $(PROTOBUF_LIB)

snapshot: CFLAGS += $(PRODUCT_FLAGS) -D_SNAPSHOT -I $(GTEST_INCLUDE)
snapshot: LINKFLAG += $(PRODUCT_FLAGS) -I $(GTEST_INCLUDE)
snapshot: $(SNAPSHOT_OBJ_FILES) cnn.pb.o
	$(CC) $^ -o $(SNAPSHOT_EXECUTABLE) $(LINKFLAG) $(DIR_PARAMS) $(TEST_LDFLAGS) $(PROTOBUF_LIB) 

snapshot_debug: CFLAGS += $(DEBUG_FLAGS) -D_SNAPSHOT -I $(GTEST_INCLUDE)
snapshot_debug: LINKFLAG += $(DEBUG_FLAGS) -I $(GTEST_INCLUDE)
snapshot_debug: $(SNAPSHOT_OBJ_FILES) cnn.pb.o
	$(CC) $^ -o $(SNAPSHOT_EXECUTABLE) $(LINKFLAG) $(DIR_PARAMS) $(TEST_LDFLAGS) $(PROTOBUF_LIB) 

-include $(addsuffix .d, $(basename $(OBJ_FILES)))

%.o: %.cpp $(PROTO_COMPILED_SRC)
	$(CC) $(CFLAGS) $(INCLUDE_STR) $(TEST_BLASFLAGS) $(PROTOBUF) -c $< -o $@
	$(CC) $(CFLAGS) $(INCLUDE_STR) $(TEST_BLASFLAGS) $(PROTOBUF) -MT $@ -MM $< > $(basename $@).d

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
	rm -f src/*.d
	rm -f tests/toprocess.bin tests/model.bin tests/model.bin.* tests/lenet_toprocess.bin tests/imgnet_toprocess.bin

