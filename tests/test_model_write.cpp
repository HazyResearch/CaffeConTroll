
#include "../src/sched/DeviceDriver_CPU.h"
#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/ConvolutionBridge.h"
#include "../src/bridges/ParallelizedBridge.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

typedef vector<AbstractBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver> *> BridgeVector;

void WriteModelToFile(const BridgeVector bridges){
  std::string filename = std::string("tests/test_write.bin");
  FILE * pFile;
  pFile = fopen (filename.c_str(), "wb");
  LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
  LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
  for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
    model = (*bridge)->get_model_cube();
    if(model){
      fwrite(model->get_p_data(), sizeof(DataType_SFFloat), model->n_elements, pFile);
    }
    bias = (*bridge)->get_bias_cube();
    if(bias){
      fwrite(bias->get_p_data(), sizeof(DataType_SFFloat), bias->n_elements, pFile);
    }
  }
  fclose(pFile);
}

void ReadModelFromFile(BridgeVector & bridges){
  std::string filename = std::string("tests/test_write.bin");
  FILE * pFile;
  pFile = fopen (filename.c_str(), "rb");
  LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
  LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
  for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
    model = (*bridge)->get_model_cube();
    if(model){
      fread(model->get_p_data(), sizeof(DataType_SFFloat), model->n_elements, pFile);
    }
    bias = (*bridge)->get_bias_cube();
    if(bias){
      fread(bias->get_p_data(), sizeof(DataType_SFFloat), bias->n_elements, pFile);
    }
  }
  fclose(pFile);
}

template <typename TypeParam>
class ReadWriteTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    ReadWriteTest() {
      data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2 = new LogicalCube<T, Layout_CRDB>(oR, oC, oD, mB);
      grad2 = new LogicalCube<T, Layout_CRDB> (oR, oC, oD, mB);

      layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
      layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

      cnn::LayerParameter layer_param;
      cnn::ConvolutionParameter * const conv_param = layer_param.mutable_convolution_param();
      conv_param->set_num_output(oD);
      conv_param->set_kernel_size(k);
      conv_param->set_pad(p);
      conv_param->set_stride(s);

      solver_param.set_base_lr(0.01);
      solver_param.set_momentum(0.0);
      solver_param.set_lr_policy("step");
      solver_param.set_stepsize(10000);

      ConvolutionBridge_ = new ParallelizedBridge<T, ConvolutionBridge<CPU_CONV_LOWERINGTYPE1,
                         TypeParam::FUNC, T, Layout_CRDB, T, Layout_CRDB, CPUDriver>, CPUDriver>(layer1,
                             layer2, &layer_param, &solver_param, &pdriver, 1, 1);
      ConvolutionBridge_->needs_to_calc_backward_grad = true;
    }

    cnn::SolverParameter solver_param;

    ParallelizedBridge<T, ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, TypeParam::FUNC, T, Layout_CRDB, T,
      Layout_CRDB, CPUDriver>, CPUDriver> * ConvolutionBridge_;

    LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;

    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    CPUDriver pdriver;

    static const int mB = 4;
    static const int iD = 3;
    static const int oD = 10;
    static const int iR = 20;
    static const int iC = 20;
    static const int k = 5;
    static const int s = 4;
    static const int p = 2;
    static const int oR = static_cast<int>((static_cast<float>(iR + 2*p - k) / s)) + 1;
    static const int oC = static_cast<int>((static_cast<float>(iC + 2*p - k) / s)) + 1;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(ReadWriteTest, DataTypes);

TYPED_TEST(ReadWriteTest, Model) {
  typedef typename TypeParam::T T;
  std::fstream input("tests/input/conv_forward_in.txt", std::ios_base::in);
  if (input.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
      input >> this->data1->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  input.close();

  std::fstream model("tests/input/conv_backward_model.txt", std::ios_base::in);
  if (model.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->oD;i++){
      model >> this->ConvolutionBridge_->get_model_cube()->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  model.close();

  std::fstream bias_file("tests/input/conv_bias_in.txt", std::ios_base::in);
  if (bias_file.is_open()){
    for(int i=0;i<this->oD;i++){
      bias_file >> this->ConvolutionBridge_->get_bias_cube()->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  bias_file.close();

  int oR = this->oR;
  int oC = this->oC;

  for (int i=0;i<oR*oC*this->oD*this->mB;i++) {
    this->data2->get_p_data()[i] = 0;
    this->grad2->get_p_data()[i] = i*0.1;
  }

  vector<AbstractBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver> *> bridges;
  bridges.push_back(this->ConvolutionBridge_);

  for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
    (*bridge)->forward();
  }

  WriteModelToFile(bridges);
  (*bridges.begin())->get_model_cube()->reset_cube();
  ReadModelFromFile(bridges);

  for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
    (*bridge)->backward();
  }


  std::fstream expected_output("tests/output/conv_backward.txt", std::ios_base::in);
  T output;
  int idx = 0;
  if (expected_output.is_open()) {
    while (expected_output >> output) {
      EXPECT_NEAR(this->grad1->get_p_data()[idx++], output, EPS);
    }
  }else{
    FAIL();
  }
  expected_output.close();

  std::fstream expected_bias("tests/output/conv_bias.txt", std::ios_base::in);

  idx = 0;
  if (expected_bias.is_open()) {
    while (expected_bias >> output) {
      EXPECT_NEAR(this->ConvolutionBridge_->get_bias_cube()->get_p_data()[idx++], output, EPS); }
  }else{
    FAIL();
  }
  expected_bias.close();

  std::fstream expected_weights("tests/output/conv_weights.txt", std::ios_base::in);
  idx = 0;
  if (expected_weights.is_open()) {
    while (expected_weights >> output) {
      EXPECT_NEAR(this->ConvolutionBridge_->get_model_cube()->get_p_data()[idx++], output, 0.1);}
  }else{
    FAIL();
  }
  expected_weights.close();
}
