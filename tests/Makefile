CC=g++
CFLAGS=-O2 -std=c++0x
LDFLAGS=-lgtest -lpthread -lglog -lrt
BLASFLAGS=-lblas -lm -I/home/shubham/Documents/research/OpenBLAS/
SOURCES = test_main.cpp test_kernel.cpp test_scanner.cpp test_bridge.cpp test_connector.cpp
# test_cube.cpp test_kernel.cpp test_connector.cpp 

EXECUTABLE=test

all: $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) $(BLASFLAGS) -o $(EXECUTABLE)

