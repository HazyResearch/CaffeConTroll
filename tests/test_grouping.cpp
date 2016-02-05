#include "gtest/gtest.h"
#include "../src/DeepNet.h"

TEST(GroupingTest, RunTest) {

  char const * a = "deepnet";
  char const * b = "train";
  char const * c = "tests/imagenet_train/solver/imagenet_solver_grouping_test.prototxt";
  char const * argv[3];
  argv[0] = a;
  argv[1] = b;
  argv[2] = c;
  std::string model_file = "NA";

  cnn::SolverParameter solver_param;
  Parser::read_proto_from_text_file(argv[2], &solver_param);

  cnn::NetParameter net_param;
  Parser::read_net_params_from_text_file(solver_param.net(), &net_param);
  Corpus * corpus = DeepNet::read_corpus_from_lmdb(net_param, true);

  BridgeVector bridges;
  DeepNet::construct_network(bridges, *corpus, net_param, solver_param);

  // First, test the execution order
  EXPECT_EQ(bridges[0]->name, "conv1"); // The dupliate here is where grouping happens
  EXPECT_EQ(bridges[1]->name, "conv1");
  EXPECT_EQ(bridges[2]->name, "relu1");
  EXPECT_EQ(bridges[3]->name, "relu1");
  EXPECT_EQ(bridges[4]->name, "pool1");
  EXPECT_EQ(bridges[5]->name, "pool1");
  EXPECT_EQ(bridges[6]->name, "norm1");
  EXPECT_EQ(bridges[7]->name, "norm1");
  EXPECT_EQ(bridges[8]->name, "conv2");
  EXPECT_EQ(bridges[9]->name, "conv2");
  EXPECT_EQ(bridges[10]->name, "relu2");
  EXPECT_EQ(bridges[11]->name, "relu2");
  EXPECT_EQ(bridges[12]->name, "pool2");
  EXPECT_EQ(bridges[13]->name, "pool2");
  EXPECT_EQ(bridges[14]->name, "norm2");
  EXPECT_EQ(bridges[15]->name, "norm2");
  EXPECT_EQ(bridges[16]->name, "conv3");
  EXPECT_EQ(bridges[17]->name, "conv3");
  EXPECT_EQ(bridges[18]->name, "relu3");
  EXPECT_EQ(bridges[19]->name, "relu3");
  EXPECT_EQ(bridges[20]->name, "conv4");
  EXPECT_EQ(bridges[21]->name, "conv4");
  EXPECT_EQ(bridges[22]->name, "relu4");
  EXPECT_EQ(bridges[23]->name, "relu4");
  EXPECT_EQ(bridges[24]->name, "conv5");
  EXPECT_EQ(bridges[25]->name, "conv5");
  EXPECT_EQ(bridges[26]->name, "relu5");
  EXPECT_EQ(bridges[27]->name, "relu5");
  EXPECT_EQ(bridges[28]->name, "pool5");
  EXPECT_EQ(bridges[29]->name, "pool5");
  EXPECT_EQ(bridges[30]->name, "FUNNEL");
  EXPECT_EQ(bridges[31]->name, "fc6");
  EXPECT_EQ(bridges[32]->name, "relu6");
  EXPECT_EQ(bridges[33]->name, "drop6");
  EXPECT_EQ(bridges[34]->name, "fc7");
  EXPECT_EQ(bridges[35]->name, "relu7");
  EXPECT_EQ(bridges[36]->name, "drop7");
  EXPECT_EQ(bridges[37]->name, "fc8");
  EXPECT_EQ(bridges[38]->name, "loss");

  // Second, test the link between layers
  EXPECT_EQ(bridges[0]->p_input_layer, bridges[1]->p_input_layer); // first group has the same input block
  EXPECT_EQ(bridges[2]->p_input_layer, bridges[0]->p_output_layer);
  EXPECT_EQ(bridges[3]->p_input_layer, bridges[1]->p_output_layer);
  EXPECT_EQ(bridges[4]->p_input_layer, bridges[2]->p_output_layer);
  EXPECT_EQ(bridges[5]->p_input_layer, bridges[3]->p_output_layer);
  EXPECT_EQ(bridges[6]->p_input_layer, bridges[4]->p_output_layer);
  EXPECT_EQ(bridges[7]->p_input_layer, bridges[5]->p_output_layer);
  EXPECT_EQ(bridges[8]->p_input_layer, bridges[6]->p_output_layer);
  EXPECT_EQ(bridges[9]->p_input_layer, bridges[7]->p_output_layer);
  EXPECT_EQ(bridges[10]->p_input_layer, bridges[8]->p_output_layer);
  EXPECT_EQ(bridges[11]->p_input_layer, bridges[9]->p_output_layer);
  EXPECT_EQ(bridges[12]->p_input_layer, bridges[10]->p_output_layer);
  EXPECT_EQ(bridges[13]->p_input_layer, bridges[11]->p_output_layer);
  EXPECT_EQ(bridges[14]->p_input_layer, bridges[12]->p_output_layer);
  EXPECT_EQ(bridges[15]->p_input_layer, bridges[13]->p_output_layer);
  EXPECT_EQ(bridges[16]->p_input_layer, bridges[14]->p_output_layer);
  EXPECT_EQ(bridges[17]->p_input_layer, bridges[15]->p_output_layer);
  EXPECT_EQ(bridges[18]->p_input_layer, bridges[16]->p_output_layer);
  EXPECT_EQ(bridges[19]->p_input_layer, bridges[17]->p_output_layer);
  EXPECT_EQ(bridges[20]->p_input_layer, bridges[18]->p_output_layer);
  EXPECT_EQ(bridges[21]->p_input_layer, bridges[19]->p_output_layer);
  EXPECT_EQ(bridges[22]->p_input_layer, bridges[20]->p_output_layer);
  EXPECT_EQ(bridges[23]->p_input_layer, bridges[21]->p_output_layer);
  EXPECT_EQ(bridges[24]->p_input_layer, bridges[22]->p_output_layer);
  EXPECT_EQ(bridges[25]->p_input_layer, bridges[23]->p_output_layer);
  EXPECT_EQ(bridges[26]->p_input_layer, bridges[24]->p_output_layer);
  EXPECT_EQ(bridges[27]->p_input_layer, bridges[25]->p_output_layer);
  EXPECT_EQ(bridges[28]->p_input_layer, bridges[26]->p_output_layer);
  EXPECT_EQ(bridges[29]->p_input_layer, bridges[27]->p_output_layer);

  EXPECT_EQ(bridges[30]->p_input_layer, bridges[28]->p_output_layer);
  EXPECT_EQ(bridges[31]->p_input_layer, bridges[30]->p_output_layer);
  EXPECT_EQ(bridges[32]->p_input_layer, bridges[31]->p_output_layer);
  EXPECT_EQ(bridges[33]->p_input_layer, bridges[32]->p_output_layer);
  EXPECT_EQ(bridges[34]->p_input_layer, bridges[33]->p_output_layer);
  EXPECT_EQ(bridges[35]->p_input_layer, bridges[34]->p_output_layer);
  EXPECT_EQ(bridges[36]->p_input_layer, bridges[35]->p_output_layer);
  EXPECT_EQ(bridges[37]->p_input_layer, bridges[36]->p_output_layer);
  EXPECT_EQ(bridges[38]->p_input_layer, bridges[37]->p_output_layer);

  // Third, test the output size of CONV layer. Then should be 1/2 of the original size
  EXPECT_EQ(bridges[0]->oD, 96/2);
  EXPECT_EQ(bridges[1]->oD, 96/2);
  EXPECT_EQ(bridges[8]->oD, 256/2);
  EXPECT_EQ(bridges[9]->oD, 256/2);
  EXPECT_EQ(bridges[16]->oD, 384/2);
  EXPECT_EQ(bridges[17]->oD, 384/2);
  EXPECT_EQ(bridges[20]->oD, 384/2);
  EXPECT_EQ(bridges[21]->oD, 384/2);
  EXPECT_EQ(bridges[24]->oD, 256/2);
  EXPECT_EQ(bridges[25]->oD, 256/2);

}
