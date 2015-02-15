#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/ConvolutionBridge.h"
#include "../src/bridges/AbstractBridge.h"
#include "../src/bridges/MaxPoolingBridge.h"
#include "../src/bridges/ReLUBridge.h"
#include "../src/bridges/ConvolutionBridge.h"
#include "../src/bridges/FullyConnectedBridge.h"
#include "../src/bridges/SoftmaxLossBridge.h"
#include "../src/bridges/DropoutBridge.h"
#include "../src/bridges/LRNBridge.h"
#include "../src/bridges/ParallelizedBridge.h"
#include "../src/Layer.h"
#include "../src/parser/corpus.h"
#include "../src/util.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <algorithm>

typedef LogicalCube<DataType_SFFloat, Layout_CRDB> LogicalCubeFloat;
typedef vector<AbstractBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> *> BridgeVector;

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
    std::vector<std::pair<float, int> > data_vector;
    for (int j = 0; j < dim; ++j) {
      data_vector.push_back(
          std::make_pair(actual_data[i * dim + j], j));
    }
    std::partial_sort(
        data_vector.begin(), data_vector.begin() + top_k,
        data_vector.end(), std::greater<std::pair<float, int> >());
    for (int k = 0; k < top_k; k++) {
      if (data_vector[k].second == static_cast<int>(expected_label[i])) {
        ++accuracy;
        break;
      }
    }
  }
  return accuracy;
}

size_t compute_conv_next_layer_dimension(const size_t R_i, const size_t K,
    const size_t padding, const size_t stride ) {
  return (R_i + 2 * padding - K) / stride + 1;
}

void construct_network(BridgeVector & bridges, const Corpus & corpus, const cnn::NetParameter & net_param) {
  size_t input_R = corpus.n_rows, input_C = corpus.n_cols, input_D = corpus.dim, B = corpus.mini_batch_size,
         last_B = corpus.last_batch_size;

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
              (prev_layer, next_layer, &layer_param, 16, 1);
        }
        break;
        {
          case cnn::LayerParameter_LayerType_INNER_PRODUCT:
            output_D = layer_param.inner_product_param().num_output();
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
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

            bridge = new ParallelizedBridge<DataType_SFFloat,
              ReLUBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
              (prev_layer, next_layer, &layer_param, 16, 1); 
        }
        break;
        {
          case cnn::LayerParameter_LayerType_LRN:
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
            bridge = new ParallelizedBridge<DataType_SFFloat,
              LRNBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
              (prev_layer, next_layer, &layer_param, 16, 1);
        }
        break;
        {
          case cnn::LayerParameter_LayerType_DROPOUT:
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
            bridge = new DropoutBridge<DataType_SFFloat, Layout_CRDB,
                   DataType_SFFloat, Layout_CRDB>(prev_layer, next_layer, &layer_param);
        }
        break;
        {
          case cnn::LayerParameter_LayerType_SOFTMAX_LOSS:
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
            LogicalCubeFloat * const labels = new LogicalCubeFloat(NULL, 1, 1, 1, B);

            bridge = new SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,
                   DataType_SFFloat, Layout_CRDB>(prev_layer, next_layer, labels);
        }
        break;
        default:
        cout << "This layer type is not supported: "<< layer_type << "!" << endl;
        assert(false);
      }

      bridges.push_back(bridge);
      input_R = output_R, input_C = output_C, input_D = output_D;
      prev_data = next_data, prev_grad = next_grad;
      prev_layer = next_layer;
    }
  }
}

Corpus read_corpus_from_lmdb(const cnn::NetParameter & net_param, const string data_binary) {
  const cnn::LayerParameter layer_param = net_param.layers(0);
  if (layer_param.type() == cnn::LayerParameter_LayerType_DATA) {
    if (layer_param.include(0).phase() == 0) { // training phase
      return Corpus(layer_param, data_binary);
    }
  }
  assert(false);
}

typedef vector<AbstractBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> *> BridgeVector;
void WriteModelToFile(const BridgeVector bridges){
  std::string filename = std::string("test_write.bin"); 
  FILE * pFile;
  pFile = fopen (filename.c_str(), "wb");
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

void ReadModelFromFile(BridgeVector & bridges){
  std::string filename = std::string("test_write.bin"); 
  FILE * pFile;
  pFile = fopen (filename.c_str(), "rb");
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

template <typename TypeParam>
class LenetTrainTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    LenetTrainTest() {
      Parser::read_proto_from_text_file("caffe_inputs/lenet_solver.prototxt", &solver_param);
      Parser::read_net_params_from_text_file(solver_param.net(), &net_param);
    }

    BridgeVector bridges;  
    cnn::NetParameter net_param;
    cnn::SolverParameter solver_param;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(LenetTrainTest, DataTypes);

TYPED_TEST(LenetTrainTest, ModelTest) {

  // Training
  const Corpus corpus = read_corpus_from_lmdb(this->net_param, "test_data.bin");
  construct_network(this->bridges, corpus, this->net_param);
  SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const softmax =
    (SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) this->bridges.back();

  AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const first =
    (AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) this->bridges.front();

  LogicalCubeFloat * const labels = softmax->p_data_labels;
  LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;

  FILE * pFile;
  pFile = fopen (corpus.filename.c_str(), "rb");
  for (size_t epoch = 0; epoch < 1; ++epoch) {
    float epoch_loss = 0.0;

    
    for (size_t batch = 0, corpus_batch_index = 0; batch < corpus.num_mini_batches - 1; ++batch,
      corpus_batch_index += corpus.mini_batch_size) {
      fread(corpus.images->p_data, sizeof(DataType_SFFloat), corpus.images->n_elements, pFile);

      float * const mini_batch = corpus.images->physical_get_RCDslice(0);
      input_data->p_data = mini_batch;

      softmax->loss = 0.0;

      // initialize labels for this mini batch
      labels->p_data = corpus.labels->physical_get_RCDslice(corpus_batch_index);
      // forward pass
      for (auto bridge = this->bridges.begin(); bridge != this->bridges.end(); ++bridge) {
        (*bridge)->forward();
      }

      for (auto bridge = this->bridges.rbegin(); bridge != this->bridges.rend(); ++bridge) {
        (*bridge)->backward();
      } 
    }
  }
  fclose(pFile);

  WriteModelToFile(this->bridges);

  //Testing
  ReadModelFromFile(this->bridges);

  pFile = fopen (corpus.filename.c_str(), "rb");

  int total_accuracy = 0;
  for (size_t batch = 0, corpus_batch_index = 0; batch < corpus.num_mini_batches - 1; ++batch,
    corpus_batch_index += corpus.mini_batch_size) {
    fread(corpus.images->p_data, sizeof(DataType_SFFloat), corpus.images->n_elements, pFile);

    float * const mini_batch = corpus.images->physical_get_RCDslice(0);
    input_data->p_data = mini_batch;

    // initialize labels for this mini batch
    labels->p_data = corpus.labels->physical_get_RCDslice(corpus_batch_index);
    // forward pass
    for (auto bridge = this->bridges.begin(); bridge != this->bridges.end(); ++bridge) {
      (*bridge)->p_input_layer->p_gradient_cube->reset_cube();
      (*bridge)->p_output_layer->p_data_cube->reset_cube();
      (*bridge)->forward();
    }
     total_accuracy += find_accuracy(labels, softmax->p_output_layer->p_data_cube);
  }

  std::fstream expected_accuracy("accuracy_train.txt", std::ios_base::in);
  double output;
  if (expected_accuracy.is_open()) {
    expected_accuracy >> output;  
    EXPECT_NEAR((1.0*total_accuracy/((corpus.num_mini_batches - 1)*corpus.mini_batch_size)), output, 0.01);
  }
  expected_accuracy.close();
  fclose(pFile);
}