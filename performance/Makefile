CC=g++-4.8
CFLAGS=-O3 -std=c++11 -Wl,--no-as-needed -g
LDFLAGS=-lrt -lpthread -L ../externals/OpenBLAS/ -lopenblas
BLASFLAGS= -lm -I ../externals/OpenBLAS/
PROTOBUF = `pkg-config --cflags --libs protobuf`
#SOURCES = ../src/util.cpp ../src/parser/cnn.pb.cc ../src/timer.cpp
#SOURCES = perf_conv.cpp ../src/util.cpp ../src/parser/cnn.pb.cc ../src/timer.cpp
#perf_pooling.cpp perf_relu.cpp perf_lrn.cpp perf_dropout.cpp perf_softmax.cpp
EXECUTABLE= perf

#all: $(SOURCES)
#	$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) $(BLASFLAGS) $(PROTOBUF) -o $(EXECUTABLE)

speedup: util.o cnn.pb.o timer.o
	$(CC) -DNPARTITION=$(NPARTITION) -DNTHREADPERPARTITION=$(NTHREADPERPARTITION) \
	$(CFLAGS) perf_speedup.cpp util.o cnn.pb.o timer.o $(LDFLAGS) $(BLASFLAGS) $(PROTOBUF) -o bin/speedup.$(NPARTITION).$(NTHREADPERPARTITION)

util.o:
	$(CC) $(CFLAGS) -c ../src/util.cpp $(LDFLAGS) $(BLASFLAGS) $(PROTOBUF) 

cnn.pb.o:
	$(CC) $(CFLAGS) -c ../src/parser/cnn.pb.cc $(LDFLAGS) $(BLASFLAGS) $(PROTOBUF)

timer.o:
	$(CC) $(CFLAGS) -c ../src/timer.cpp $(LDFLAGS) $(BLASFLAGS) $(PROTOBUF)

clean:
	rm *.o
