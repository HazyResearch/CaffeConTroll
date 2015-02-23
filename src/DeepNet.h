#ifndef _moka_DeepNet_h
#define _moka_DeepNet_h

#include <iostream>
#include <algorithm>
#include <boost/program_options.hpp>
#include "config.h"
#include "LogicalCube.h"
#include "bridges/AbstractBridge.h"
#include "util.h"

typedef LogicalCube<DataType_SFFloat, Layout_CRDB> LogicalCubeFloat;
typedef AbstractBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> Bridge;
typedef std::vector<Bridge *> BridgeVector;

class DeepNet {
  protected:
    static bool train_;
  public:
    static bool train() { return train_; }

    static void load_and_train_network(const char * file, const std::string data_binary, const std::string model_file);

    static float load_and_test_network(const char * file, const std::string data_binary, const std::string model_file);

    static Corpus * load_network(const char * file, const string & data_binary, cnn::SolverParameter & solver_param,
        cnn::NetParameter & net_param, BridgeVector & bridges, bool train);

    static Corpus * read_corpus_from_lmdb(const cnn::NetParameter & net_param, const std::string data_binary, bool train);

    static void construct_network(BridgeVector & bridges, Corpus & corpus, const cnn::NetParameter & net_param,
        const cnn::SolverParameter & solver_param);

    static int find_accuracy(const LogicalCubeFloat * const labels, const LogicalCubeFloat * output);
};

#include "Layer.h"
#include "Connector.h"
#include "Kernel.h"
#include "bridges/MaxPoolingBridge.h"
#include "bridges/ReLUBridge.h"
#include "bridges/ConvolutionBridge.h"
#include "bridges/FullyConnectedBridge.h"
#include "bridges/SoftmaxLossBridge.h"
#include "bridges/DropoutBridge.h"
#include "bridges/LRNBridge.h"
#include "bridges/ParallelizedBridge.h"
#include "bridges/FunnelBridge.h"
#include "parser/corpus.h"

#endif
