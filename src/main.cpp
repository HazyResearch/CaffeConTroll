//
//  main.cpp
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include <iostream>
#include <boost/program_options.hpp>
#include "config.h"
#include "LogicalCube.h"
#include "Connector.h"
#include "Kernel.h"
#include "bridges/AbstractBridge.h"
#include "bridges/MaxPoolingBridge.h"
#include "bridges/ReLUBridge.h"
#include "bridges/ConvolutionBridge.h"
#include "bridges/FullyConnectedBridge.h"
#include "bridges/SoftmaxLossBridge.h"
#include "bridges/DropoutBridge.h"
#include "bridges/LRNBridge.h"
#include "bridges/ParallelizedBridge.h"
#include "Layer.h"
#include "parser/corpus.h"
#include "util.h"
#include <algorithm>

using namespace std;

typedef LogicalCube<DataType_SFFloat, Layout_CRDB> LogicalCubeFloat;
typedef vector<AbstractBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> *> BridgeVector;

// computes the output dimension for any convolution layer
inline size_t compute_conv_next_layer_dimension(const size_t R_i, const size_t K,
    const size_t padding, const size_t stride ) {
  return (R_i + 2 * padding - K) / stride + 1;
}

// load training data into Corpus object, return Corpus object
// Note: we assume that the very first layer in the .protoxt
// file specifies the data layer
// TODO: also read in test set
Corpus read_corpus_from_lmdb(const cnn::NetParameter & net_param, const string data_binary, bool train) {
  if (train){
    const cnn::LayerParameter layer_param = net_param.layers(0);
    if (layer_param.type() == cnn::LayerParameter_LayerType_DATA) {
      if (layer_param.include(0).phase() == 0) { // training phase
        return Corpus(layer_param, data_binary);
      }
    }  
  }
  else{
    const cnn::LayerParameter layer_param = net_param.layers(1);
    if (layer_param.type() == cnn::LayerParameter_LayerType_DATA) {
      if (layer_param.include(0).phase() == 1) { // training phase
        return Corpus(layer_param, data_binary);
      }
    }  
  }
  
  cout << "No data layer present in prototxt file!" << endl;
  assert(false);
}

//// Shubham: Need to be refactored a bit on the basis of how these features would actually be used.
/// Should we have a separate test function?
void WriteModelToFile(const BridgeVector bridges, const string model_file){
  FILE * pFile;
  pFile = fopen (model_file.c_str(), "wb");
  LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
  LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
  for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
    model = (*bridge)->get_model_cube();
    if(model){
      fwrite (model->p_data , sizeof(DataType_SFFloat), model->n_elements, pFile);  
    }
    bias = (*bridge)->get_bias_cube();
    if(bias){
      fwrite (bias->p_data , sizeof(DataType_SFFloat), bias->n_elements, pFile); 
    }
  }
  fclose(pFile);
}

void ReadModelFromFile(BridgeVector & bridges, const string model_file){
  FILE * pFile;
  pFile = fopen (model_file.c_str(), "rb");
  LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
  LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
  for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
    model = (*bridge)->get_model_cube();
    if(model){
      fread(model->p_data , sizeof(DataType_SFFloat), model->n_elements, pFile);  
    }
    bias = (*bridge)->get_bias_cube();
    if(bias){
      fread(bias->p_data , sizeof(DataType_SFFloat), bias->n_elements, pFile); 
    }
  }
  fclose(pFile);
}

int find_accuracy(const LogicalCubeFloat * const labels, const LogicalCubeFloat * output) {
  const float* actual_data = output->p_data;
  const float* expected_label = labels->p_data;
  int top_k = 1;
  float accuracy = 0;
  int num = output->B;
  int dim = output->D;
  vector<float> maxval(top_k+1);
  vector<int> max_id(top_k+1);
  for (int i = 0; i < num; ++i) {
    // Top-k accuracy
    std::vector<std::pair<float, int> > data_vector;
    for (int j = 0; j < dim; ++j) {
      data_vector.push_back(
          std::make_pair(actual_data[i * dim + j], j));
    }
    std::partial_sort(
        data_vector.begin(), data_vector.begin() + top_k,
        data_vector.end(), std::greater<std::pair<float, int> >());
    // check if true label is in top k predictions
    for (int k = 0; k < top_k; k++) {
      if (data_vector[k].second == static_cast<int>(expected_label[i])) {
        ++accuracy;
        break;
      }
    }
  }
  return accuracy;
  //cout << "Accuracy: " << (accuracy / num) << endl;
}

// This takes in the bridge vector (which has been initialized to be empty in load_and_train_network)
// and builds up a list of bridges in the vector in the order in which they will be executed in the forward
// pass. Only the bridges variable is modified.
void construct_network(BridgeVector & bridges, const Corpus & corpus, const cnn::NetParameter & net_param) {
  size_t input_R = corpus.n_rows, input_C = corpus.n_cols, input_D = corpus.dim, B = corpus.mini_batch_size,
         last_B = corpus.last_batch_size;

  // Create the Logical Cubes for the initial data layer
  LogicalCubeFloat * prev_data = new LogicalCubeFloat(corpus.images->physical_get_RCDslice(0), input_R, input_C, input_D, B);
  LogicalCubeFloat * prev_grad = new LogicalCubeFloat(input_R, input_C, input_D, B);

  Layer<DataType_SFFloat, Layout_CRDB> * prev_layer = new Layer<DataType_SFFloat, Layout_CRDB>(prev_data, prev_grad);

  const size_t num_layers = net_param.layers_size();

  AbstractBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> * bridge = NULL;
  LogicalCube<DataType_SFFloat, Layout_CRDB> * next_data = NULL;
  LogicalCube<DataType_SFFloat, Layout_CRDB> * next_grad = NULL;
  Layer<DataType_SFFloat, Layout_CRDB> * next_layer = NULL;

  size_t output_R = input_R, output_C = input_C, output_D = input_D;

  for (size_t i = 1; i < num_layers; ++i) {

    const cnn::LayerParameter layer_param = net_param.layers(i);
    const cnn::LayerParameter_LayerType layer_type = layer_param.type();

    if (layer_type != cnn::LayerParameter_LayerType_DATA) {
      switch (layer_type) {
        // Note: These braces surrounding each case statement are necessary
        // because we're initializing variables (such as config) inside the case.
        // (Otherwise, the compiler will complain about a "switch case is in protected
        // scope" error.)
        {
          case cnn::LayerParameter_LayerType_CONVOLUTION:
            const size_t K = layer_param.convolution_param().kernel_size(),
                  padding = layer_param.convolution_param().pad(),
                  stride = layer_param.convolution_param().stride();

            output_R = compute_conv_next_layer_dimension(input_R, K, padding, stride),
                     output_C = compute_conv_next_layer_dimension(input_C, K, padding, stride),
                     output_D = layer_param.convolution_param().num_output();

            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

            bridge = new ParallelizedBridge<DataType_SFFloat,
              ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
              (prev_layer, next_layer, &layer_param, 16, 1); // TODO: need a CMD line option here -- but currently we do not have the interface to do that.
        }
        break;
        {
          case cnn::LayerParameter_LayerType_INNER_PRODUCT:
            output_D = layer_param.inner_product_param().num_output();

            // The R and C dimensions for a fully connected layer are always 1 x 1
            output_R = output_C = 1;
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

            bridge = new FullyConnectedBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>(prev_layer,
            next_layer, &layer_param);
        }
        break;
        {
          case cnn::LayerParameter_LayerType_POOLING:
            const size_t K = layer_param.pooling_param().kernel_size(), stride = layer_param.pooling_param().stride();

            output_R = compute_conv_next_layer_dimension(input_R, K, 0, stride),
                     output_C = compute_conv_next_layer_dimension(input_C, K, 0, stride);

            // input_D same as output_D
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, input_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, input_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

            bridge = new ParallelizedBridge<DataType_SFFloat,
              MaxPoolingBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >(prev_layer,
                  next_layer, &layer_param, 16, 1);

        }
        break;
        {
          case cnn::LayerParameter_LayerType_RELU:
            // input_[R,C,D] is the same as output_[R,C,D]
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

            bridge = new ParallelizedBridge<DataType_SFFloat,
              ReLUBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
              (prev_layer, next_layer, &layer_param, 16, 1); // TODO: need a CMD line option here -- but currently we do not have the interface to do that.

            /*
            bridge = new ReLUBridge<DataType_SFFloat, Layout_CRDB,
                   DataType_SFFloat, Layout_CRDB>(prev_layer, next_layer, &layer_param);
            */
        }
        break;
        {
          case cnn::LayerParameter_LayerType_LRN:
            // input_[R,C,D] is the same as output_[R,C,D]
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
            //bridge = new ParallelizedLRNBridge<DataType_SFFloat>(prev_layer, next_layer, &layer_param, 4, 2);
            //bridge = new LRNBridge<DataType_SFFloat, Layout_CRDB,
            //       DataType_SFFloat, Layout_CRDB>(prev_layer, next_layer, &layer_param);
            bridge = new ParallelizedBridge<DataType_SFFloat,
              LRNBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
              (prev_layer, next_layer, &layer_param, 16, 1); // TODO: need a CMD line option here -- but currently we do not have the interface to do that.

        }
        break;
        {
          case cnn::LayerParameter_LayerType_DROPOUT:
            // input_[R,C,D] is the same as output_[R,C,D]
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
            bridge = new DropoutBridge<DataType_SFFloat, Layout_CRDB,
                   DataType_SFFloat, Layout_CRDB>(prev_layer, next_layer, &layer_param);
        }
        break;
        {
          case cnn::LayerParameter_LayerType_SOFTMAX_LOSS:
            // input_[R,C,D] is the same as output_[R,C,D]
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
            // must be initialized to point to next mini batch
            LogicalCubeFloat * const labels = new LogicalCubeFloat(NULL, 1, 1, 1, B);

            bridge = new SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,
                   DataType_SFFloat, Layout_CRDB>(prev_layer, next_layer, labels);
        }
        break;
        default:
        cout << "This layer type is not supported: "<< layer_type << "!" << endl;
        assert(false);
      }

      // Appending the bridge to our vector of bridges, and updating pointers
      // and values for the next iteration.
      bridges.push_back(bridge);
      input_R = output_R, input_C = output_C, input_D = output_D;
      prev_data = next_data, prev_grad = next_grad;
      prev_layer = next_layer;
      //ReadModelFromFile(bridges);
    }
  }
}

// Here, we train our CNN: we iterate over the vector of bridges, forwards and backward for each batch size.
// Right now, we do this in a single-thread fashion. TODO: Create a Scheduler class, that schedules workers
// for each batch size, so that we can perform these forward and backward passes in parallel.
void train_network(const BridgeVector & bridges, const Corpus & corpus, const cnn::NetParameter & net_param,
    const cnn::SolverParameter & solver_param) {

  // TODO: we need a more general AbstractLossBridge
  SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const softmax =
    (SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) bridges.back();

  AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const first =
    (AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) bridges.front();

  LogicalCubeFloat * const labels = softmax->p_data_labels;
  LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;

  const size_t num_epochs = solver_param.max_iter();
  Timer t = Timer();
  for (size_t epoch = 0; epoch < 1; ++epoch) {
    cout << "EPOCH: " << epoch << endl;
    float epoch_loss = 0.0;

    FILE * pFile;
    pFile = fopen (corpus.filename.c_str(), "rb");

    // num_mini_batches - 1, because we need one more iteration for the final mini batch
    // (the last mini batch may not be the same size as the rest of the mini batches)
    for (size_t batch = 0, corpus_batch_index = 0; batch < corpus.num_mini_batches - 1; ++batch,
        corpus_batch_index += corpus.mini_batch_size) {
      cout << "BATCH: " << batch << endl;

      Timer t;

      // this loading appears to take just ~ 0.1 s for each batch,
      // so double-buffering seems an overkill here because the following operations took seconds...
      fread(corpus.images->p_data, sizeof(DataType_SFFloat), corpus.images->n_elements, pFile);
      std::cout << "loading elpased " << t.elapsed() << std::endl;
      t.restart();

      // initialize input_data for this mini batch
      // Ce: Notice the change here compared with the master branch -- this needs to be refactored
      // to make the switching between this and the master branch (that load everything in memory)
      // dynamically and improve code reuse.
      float * const mini_batch = corpus.images->physical_get_RCDslice(0);
      input_data->p_data = mini_batch;

      softmax->loss = 0.0;

      // initialize labels for this mini batch
      labels->p_data = corpus.labels->physical_get_RCDslice(corpus_batch_index);

      // forward pass
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        // Reset gradient and data cubes for backward and forward passes, respectively,
        // since we don't want any leftover values from the previous iteration
        //(*bridge)->p_input_layer->p_gradient_cube->reset_cube();
        //(*bridge)->p_output_layer->p_data_cube->reset_cube();
        (*bridge)->forward();
        (*bridge)->report_forward_last_transfer.print();
      }
      std::cout << "fwd elpased " << t.elapsed() << std::endl;
      t.restart();

      cout << "LOSS: " << (softmax->loss / corpus.mini_batch_size) << endl;
      epoch_loss += (softmax->loss / corpus.mini_batch_size);
      find_accuracy(labels, (*--bridges.end())->p_output_layer->p_data_cube);

      // backward pass
      for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
        (*bridge)->backward();
      }
    }

    fclose(pFile);

    // TODO: handle the very last batch, which may not have the same
    // batch size as the rest of the batches
    cout << "Time Elapsed for a single epoch: " << t.elapsed() << endl;
  }
  cout << "Total Time Elapsed: " << t.elapsed() << endl;

}

void test_network(const BridgeVector & bridges, const Corpus & corpus, const cnn::NetParameter & net_param,
    const cnn::SolverParameter & solver_param) {

  // TODO: we need a more general AbstractLossBridge
  SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const softmax =
    (SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) bridges.back();

  AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const first =
    (AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) bridges.front();

  LogicalCubeFloat * const labels = softmax->p_data_labels;
  LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;

  FILE * pFile;
  pFile = fopen (corpus.filename.c_str(), "rb");

  // num_mini_batches - 1, because we need one more iteration for the final mini batch
  // (the last mini batch may not be the same size as the rest of the mini batches)
  int batch_accuracy;
  int total_accuracy = 0; 
  for (size_t batch = 0, corpus_batch_index = 0; batch < corpus.num_mini_batches - 1; ++batch,
      corpus_batch_index += corpus.mini_batch_size) {
    cout << "BATCH: " << batch << endl;
    fread(corpus.images->p_data, sizeof(DataType_SFFloat), corpus.images->n_elements, pFile);

    float * const mini_batch = corpus.images->physical_get_RCDslice(0);
    input_data->p_data = mini_batch;

    softmax->loss = 0.0;

    // initialize labels for this mini batch
    labels->p_data = corpus.labels->physical_get_RCDslice(corpus_batch_index);
    // forward pass
    for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
      (*bridge)->p_input_layer->p_gradient_cube->reset_cube();
      (*bridge)->p_output_layer->p_data_cube->reset_cube();
      (*bridge)->forward();
    }

    cout << "LOSS: " << (softmax->loss / corpus.mini_batch_size) << endl;
    batch_accuracy = find_accuracy(labels, softmax->p_output_layer->p_data_cube);
    cout << "Batch" << batch << " Accuracy " << batch_accuracy << endl;
    total_accuracy += batch_accuracy;
  }
  cout << "Total Accuracy" << (1.0*total_accuracy/((corpus.num_mini_batches - 1)*corpus.mini_batch_size)) << endl;
  fclose(pFile);
}

// We expect this to be called from main,
// it takes in a const char * argument (most likely
// from arvg[1]) that represents the .prototxt file
// which specifies the *solver* for the network, not
// the network configuration file itself.
//
// There are three steps involved in training the network:
//
// 1) Load the necessary training data into a Corpus object,
//    which will contain both the data itself, and the correct
//    labels.
//
// 2) Construct the necessary Bridge, Layer, and LogicalCube
//    objects to represent the network. A network should be
//    represented as an STL vector of Bridge pointers, so that we
//    can easily compute the forward pass and the backward pass.
//
// 3) For epoch = 0 -> num_epochs (<- extracted from prototxt file)
//      For batch = 0 -> num_batches - 1 (<- extracted from protoxt file)
//        Compute forward pass (iterate through vector of Bridge pointers)
//      Compute forward pass for last batch (might not have the same
//                                           size as the rest of batches)
//      For batch = 0 -> num_batches - 1 (<- extracted from protoxt file)
//        Compute backward pass (iterate through vector of Bridge
//                               pointers backwards)
//      Compute backward pass for last batch (again, might not have the same
//                                            size as the rest of batches)
//
void load_and_train_network(const char * file, const string data_binary, const string model_file) {
  // Step 1:
  cnn::SolverParameter solver_param;
  Parser::read_proto_from_text_file(file, &solver_param);

  cnn::NetParameter net_param;
  Parser::read_net_params_from_text_file(solver_param.net(), &net_param);
  const Corpus corpus = read_corpus_from_lmdb(net_param, data_binary, true);

#ifdef _DO_WARNING
  cout << "Corpus train loaded" << endl;
  cout << "CORPUS NUM IMAGES: " << corpus.n_images << endl;
  cout << "CORPUS NUM ROWS: " << corpus.n_rows << endl;
  cout << "CORPUS NUM COLS: " << corpus.n_cols << endl;
  cout << "CORPUS NUM CHANNELS: " << corpus.dim << endl;
  cout << "CORPUS MINI BATCH SIZE: " << corpus.mini_batch_size << endl;
  cout << "CORPUS NUM MINI BATCHES: " << corpus.num_mini_batches << endl;
  cout << "CORPUS LAST BATCH SIZE: " << corpus.last_batch_size << endl;
#endif

  // Step 2:
  BridgeVector bridges;
  construct_network(bridges, corpus, net_param);

  // Step 3:
  // Now, the bridges vector is fully populated
  train_network(bridges, corpus, net_param, solver_param);
  if(model_file == "NA")
    WriteModelToFile(bridges, "deepnetmodel.bin");  
  else
    WriteModelToFile(bridges, model_file);
  // Step 4:
  // Clean up! TODO: free the allocated bridges, layers, and cubes
}

void load_and_test_network(const char * file, const string data_binary, const string model_file) {
  // Step 1:
  cnn::SolverParameter solver_param;
  Parser::read_proto_from_text_file(file, &solver_param);

  cnn::NetParameter net_param;
  Parser::read_net_params_from_text_file(solver_param.net(), &net_param);
  const Corpus corpus = read_corpus_from_lmdb(net_param, data_binary, false);

#ifdef _DO_WARNING
  cout << "Corpus train loaded" << endl;
  cout << "CORPUS NUM IMAGES: " << corpus.n_images << endl;
  cout << "CORPUS NUM ROWS: " << corpus.n_rows << endl;
  cout << "CORPUS NUM COLS: " << corpus.n_cols << endl;
  cout << "CORPUS NUM CHANNELS: " << corpus.dim << endl;
  cout << "CORPUS MINI BATCH SIZE: " << corpus.mini_batch_size << endl;
  cout << "CORPUS NUM MINI BATCHES: " << corpus.num_mini_batches << endl;
  cout << "CORPUS LAST BATCH SIZE: " << corpus.last_batch_size << endl;
#endif

  // Step 2:
  BridgeVector bridges;
  construct_network(bridges, corpus, net_param);

  if(model_file != "NA"){
    ReadModelFromFile(bridges, model_file); 
    test_network(bridges, corpus, net_param, solver_param); 
  }
  else{
    cout << "No Model File Provided" << endl;
  }  
}

int main(int argc, const char * argv[]) {
  if (argc < 3) {
    cout << "Usage: ./deepnet <train/test> <solver.prototxt>" << endl;
    exit(1);
  }

  string data_binary;
  string model_file;

  boost::program_options::options_description desc("Options for my program");
  desc.add_options()
      // Option 'data-binary' and 'b' are equivalent.
      ("data-binary,b", boost::program_options::value<string>(& data_binary)->default_value("toprocess.bin"),
          "Processed data binary")
      // Option 'model' and 'm' are equivalent.
      ("model,m", boost::program_options::value<string>(& model_file)->default_value(false),
          "Model binary")
      ;

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if(string(argv[1]) == "train")
    load_and_train_network(argv[2], data_binary, model_file);
  else if(string(argv[1]) == "test")
    load_and_test_network(argv[2], data_binary, model_file);
  
  return 0;
}
