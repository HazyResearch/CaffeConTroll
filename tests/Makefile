include ../.config

CFLAGS=-O0 -std=c++11 -g -I $(GTEST_INCLUDE)
LDFLAGS=-lgtest -lpthread  -L $(OPENBLAS_DIR) -L$(GTEST_LIB_DIR) -lopenblas

ifeq ($(UNAME), Darwin)
  CFLAGS = $(CFLAGS) -Wall -std=c++11
else ifeq ($(UNAME), Linux)
  CFLAGS = $(CFLAGS) -Wall -std=c++11 -Wl,--no-as-needed
  LDFLAGS = $(LDFLAGS) -lrt
endif

BLASFLAGS= -lm -I $(OPENBLAS_DIR) 
PROTOBUF = `pkg-config --cflags --libs protobuf`
SOURCES   = test_main.cpp test_MaxPooling_bridge.cpp test_softmax_bridge.cpp ../src/util.cpp ../src/parser/cnn.pb.cc ../src/timer.cpp
EXECUTABLE=test

all: $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) $(LDFLAGS) $(BLASFLAGS) $(PROTOBUF) -o $(EXECUTABLE) 

