CC=g++-4.8
CFLAGS=-O0 -std=c++11 -g 
LDFLAGS=-lgtest -lpthread  -L ../externals/OpenBLAS/ -lopenblas
BLASFLAGS= -lm -I ../externals/OpenBLAS/ 
PROTOBUF = `pkg-config --cflags --libs protobuf`
SOURCES = test_main.cpp test_MaxPooling_bridge.cpp test_softmax_bridge.cpp ../src/util.cpp ../src/parser/cnn.pb.cc ../src/timer.cpp
#SOURCES = test_main.cpp test_MaxPooling_bridge.cpp test_softmax_bridge.cpp test_convolution_bridge.cpp test_ReLU_bridge.cpp test_dropout_bridge.cpp test_lrn_bridge.cpp test_parallelized_convolution.cpp ../src/util.cpp ../src/parser/cnn.pb.cc ../src/timer.cpp
#test_MaxPooling_bridge.cpp test_softmax_bridge.cpp test_convolution_bridge.cpp test_ReLU_bridge.cpp test_dropout_bridge.cpp test_lrn_bridge.cpp
# test_convolution_bridge.cpp test_ReLU_bridge.cpp test_dropout_bridge.cpp test_lrn_bridge.cpp  
#test_softmax_bridge.cpp 
#test_ReLU_bridge.cpp test_convolution_bridge.cpp 
# test_cube.cpp test_kernel.cpp test_connector.cpp 
#test_scanner.cpp test_kernel.cpp test_connector.cpp test_bridge.cpp
EXECUTABLE=test

all: $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) $(BLASFLAGS) $(PROTOBUF) -o $(EXECUTABLE) -lrt

