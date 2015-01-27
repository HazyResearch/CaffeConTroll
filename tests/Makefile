CC=g++
CFLAGS=-O2 -std=c++0x
LDFLAGS=-lgtest -lpthread -lrt -L/home/shubham/Documents/research/OpenBLAS/ -lopenblas
BLASFLAGS=-lblas -lm -I/home/shubham/Documents/research/OpenBLAS/
SOURCES = test_main.cpp test_convolution_bridge.cpp test_MaxPooling_bridge.cpp test_ReLU_bridge.cpp
#test_convolution_bridge.cpp 
# test_cube.cpp test_kernel.cpp test_connector.cpp 
#test_scanner.cpp test_kernel.cpp test_connector.cpp test_bridge.cpp
EXECUTABLE=test

all: $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) $(BLASFLAGS) -o $(EXECUTABLE)

