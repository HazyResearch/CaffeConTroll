#include "../src/DeepNet.h"
#include "gtest/gtest.h"
#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/DropoutBridge.h"
#include "test_types.h"


template <typename TypeParam>
class LenetTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    LenetTest() {
      Parser::read_proto_from_text_file("caffe_inputs/lenet_solver.prototxt", &solver_param);
      Parser::read_net_params_from_text_file(solver_param.net(), &net_param);
    }

    BridgeVector bridges;  
    cnn::NetParameter net_param;
    cnn::SolverParameter solver_param;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(LenetTest, DataTypes);

TYPED_TEST(LenetTest, RunTest) {

  cnn::SolverParameter solver_param;

  Corpus * corpus = read_corpus_from_lmdb(this->net_param, "tests/test_data.bin", false);
  construct_network(this->bridges, *corpus, this->net_param, solver_param);
  SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const softmax =
    (SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) this->bridges.back();

  AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const first =
    (AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) this->bridges.front();

  LogicalCubeFloat * const labels = softmax->p_data_labels;
  LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;

  int total_accuracy = 0;
  for (size_t epoch = 0; epoch < 1; ++epoch) {
    float epoch_loss = 0.0;

    FILE * pFile;
    pFile = fopen (corpus->filename.c_str(), "rb");

    for (size_t batch = 0, corpus_batch_index = 0; batch < corpus->num_mini_batches - 1; ++batch,
      corpus_batch_index += corpus->mini_batch_size) {
      fread(corpus->images->p_data, sizeof(DataType_SFFloat), corpus->images->n_elements, pFile);

      float * const mini_batch = corpus->images->physical_get_RCDslice(0);
      input_data->p_data = mini_batch;

      softmax->reset_loss();

      // initialize labels for this mini batch
      labels->p_data = corpus->labels->physical_get_RCDslice(corpus_batch_index);
      // forward pass
      for (auto bridge = this->bridges.begin(); bridge != this->bridges.end(); ++bridge) {
        (*bridge)->p_input_layer->p_gradient_cube->reset_cube();
        (*bridge)->p_output_layer->p_data_cube->reset_cube();
        (*bridge)->forward();
      }
       total_accuracy += find_accuracy(labels, softmax->p_output_layer->p_data_cube);
    }

    std::fstream expected_accuracy("tests/accuracy_train.txt", std::ios_base::in);
    double output;
    if (expected_accuracy.is_open()) {
      expected_accuracy >> output;  
      EXPECT_NEAR((1.0*total_accuracy/((corpus->num_mini_batches - 1)*corpus->mini_batch_size)), output, 0.01);
    }else{
      FAIL();
    }
    expected_accuracy.close();
    fclose(pFile);
  }
}