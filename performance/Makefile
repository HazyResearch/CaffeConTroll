CC=g++-4.8
CFLAGS=-O0 -std=c++11 -Wl,--no-as-needed -g
LDFLAGS=-lrt -lpthread -L ../externals/OpenBLAS/ -lopenblas
BLASFLAGS= -lm -I ../externals/OpenBLAS/
PROTOBUF = `pkg-config --cflags --libs protobuf`
SOURCES = perf_conv.cpp ../src/util.cpp ../src/parser/cnn.pb.cc
#perf_pooling.cpp perf_relu.cpp perf_lrn.cpp perf_dropout.cpp perf_softmax.cpp
EXECUTABLE= perf

all: $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) $(BLASFLAGS) $(PROTOBUF) -o $(EXECUTABLE)

