CC=g++-4.8
CFLAGS=-O2 -std=c++11 -Wl,--no-as-needed -g
LDFLAGS=-lgtest -lpthread -lrt -L ../externals/OpenBLAS/ -lopenblas
BLASFLAGS= -lm -I ../externals/OpenBLAS/
SOURCES = test_main.cpp test_MaxPooling_bridge.cpp ../src/util.cpp
#test_softmax_bridge.cpp 
#test_ReLU_bridge.cpp test_convolution_bridge.cpp 
# test_cube.cpp test_kernel.cpp test_connector.cpp 
#test_scanner.cpp test_kernel.cpp test_connector.cpp test_bridge.cpp
EXECUTABLE=test

all: $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) $(BLASFLAGS) -o $(EXECUTABLE)

