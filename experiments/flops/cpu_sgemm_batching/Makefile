include ../../../.config

LIBS=$(BLAS_LIBS) 
LD_BASE=$(foreach l, $(LIBS), -l$l)

INCLUDE_DIRS=$(BLAS_INCLUDE)
INCLUDE_STR=$(foreach d, $(INCLUDE_DIRS), -I$d)

LIB_DIRS=$(BLAS_LIB_DIR)
LIB_STR=$(foreach d, $(LIB_DIRS), -L$d)

CFLAGS = -Wall -Ofast $(BLAS_DEFS)
LDFLAGS = $(LD_BASE) -lpthread
NVCCFLAGS = -D_GPU_TARGET -D_INCLUDE_GPUDRIVER -std=c++11 $(LD_BASE) -lcublas -lcuda -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 -I $(CUDA_INCLUDE) -L $(CUDA_LIB)

TARGET = sgemm_batching
SRC = sgemm_batching.cpp
OBJ_FILES = $(patsubst %.cpp,%.o,$(SRC))

ifdef NVCC
MAIN_CUDA_SOURCES = sgemm_batching.cu
MAIN_CUDA_OBJ_FILES = $(patsubst %.cu,%.o,$(MAIN_CUDA_SOURCES))
CFLAGS += -D_INCLUDE_GPUDRIVER  -I $(CUDA_INCLUDE) -L $(CUDA_LIB)
endif

LINKCC = $(CC)
LINKFLAG = $(CFLAGS) $(LDFLAGS)

ifdef NVCC
LINKFLAG += -lcublas -lcudart
NVCC_LINK = dlink.o
endif

.PHONY: all clean

all: $(OBJ_FILES) $(MAIN_CUDA_OBJ_FILES)
ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TARGET) $(LINKFLAG) $(INCLUDE_STR) $(LIB_STR) $(LDFLAGS)
	rm -f $(TARGET).o

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDE_STR) -c $< -o $@

%.o: %.cu
	$(NVCC) -O3 $(BLAS_DEFS) $(NVCCFLAGS) $(INCLUDE_STR) -dc $< -o $@

clean:
	rm -f $(TARGET)
