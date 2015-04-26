

#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/util.h"
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

#include "../src/sched/DeviceDriver_GPU.h"
#include "../src/kernels/lowering.h"


template <typename T>
void simple_conv(LogicalCube<T, Layout_CRDB>* in, LogicalCube<T, Layout_CRDB>* kernel, LogicalCube<T, Layout_CRDB>* out){
  int ofm = out->D;
  int ifm = in->D;
  for (int n = 0; n < out->B; n++) {
    for (int o = 0; o < ofm; o++) {
      for (int k = 0; k < ifm; k++) {
        for (int y = 0; y < out->R; y++) {
          for (int x = 0; x < out->C; x++) {
            for (int p = 0; p < kernel->R; p++) {
              for (int q = 0; q < kernel->C; q++) {
                int in_y = y + p;
                int in_x = x + q;
                *out->logical_get(y, x, o, n) +=
                  *in->logical_get(in_y, in_x, k, n)*
                  *kernel->logical_get(p, q, k, o);
              }
            }
          }
        }
      }
    }
  }
}

template <typename TypeParam>
class ParallelizedConvolutionBridgeTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    ParallelizedConvolutionBridgeTest(){
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

      /***
      ParallelizedConvolutionBridge_ = new ParallelizedBridge<DataType_SFFloat,
              ConvolutionBridge<DataType_SFFloat,
              Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
              (layer1, layer2, &layer_param, &solver_param, 1, 1);
      ****/
      ParallelizedConvolutionBridge_ = new ConvolutionBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>
              (layer1, layer2, &layer_param, &solver_param);

      //ParallelizedConvolutionBridge_->p_driver = new GPUDriver();

      ParallelizedConvolutionBridge_->needs_to_calc_backward_grad = true;
    }

    virtual ~ParallelizedConvolutionBridgeTest() { delete layer1; delete layer2; }
    
    //ParallelizedBridge<DataType_SFFloat,
    //          ConvolutionBridge<DataType_SFFloat,
    //          Layout_CRDB, DataType_SFFloat, Layout_CRDB> >* ParallelizedConvolutionBridge_;

    ConvolutionBridge<DataType_SFFloat,
      Layout_CRDB, DataType_SFFloat, Layout_CRDB> * ParallelizedConvolutionBridge_;

    LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;

    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    cnn::SolverParameter solver_param;

    /*
    static const int mB = 4;
    static const int iD = 3;
    static const int oD = 10;
    static const int iR = 20;
    static const int iC = 20;
    static const int k = 5;
    static const int s = 2;
    static const int p = 2;
    */

    static const int mB = 256;
    static const int iD = 48;
    static const int oD = 128;
    static const int iR = 27;
    static const int iC = 27;
    static const int k = 5;
    static const int s = 1;
    static const int p = 2;

    static const int oR = static_cast<int>((static_cast<float>(iR + 2*p - k) / s)) + 1;
    static const int oC = static_cast<int>((static_cast<float>(iC + 2*p - k) / s)) + 1;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(ParallelizedConvolutionBridgeTest, DataTypes);

TYPED_TEST(ParallelizedConvolutionBridgeTest, TestInitialization){
  EXPECT_TRUE(this->ParallelizedConvolutionBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}


TYPED_TEST(ParallelizedConvolutionBridgeTest, TestForward){

  /*
  std::fstream input("tests/input/conv_forward_in.txt", std::ios_base::in);
  if (input.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
      input >> this->data1->get_p_data()[i];
      this->grad1->get_p_data()[i] = 0;
    }
  }
  else{
    FAIL();
  }
  input.close();
  */

  /*
  std::fstream model("tests/input/conv_model.txt", std::ios_base::in);
  if (model.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->oD;i++){
      model >> this->ParallelizedConvolutionBridge_->p_model_cube->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  model.close();

  std::fstream bias_file("tests/input/conv_bias_in.txt", std::ios_base::in);
  if (bias_file.is_open()){
    for(int i=0;i<this->oD;i++){
      bias_file >> this->ParallelizedConvolutionBridge_->p_bias_cube->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  bias_file.close();
  */

  this->ParallelizedConvolutionBridge_->forward();

  this->ParallelizedConvolutionBridge_->report_forward_last_transfer.print();
  this->ParallelizedConvolutionBridge_->report_forward_kernel.print();
  this->ParallelizedConvolutionBridge_->report_forward_lowering.print();

  /*
  std::fstream expected_output("tests/output/conv_forward.txt", std::ios_base::in);
  if(TypeParam::FUNC == FUNC_NOFUNC){
    float output;
    int idx = 0;
    if (expected_output.is_open()) {

      while (expected_output >> output){
       // EXPECT_NEAR(this->data2->get_p_data()[idx++], output, EPS);
        std::cout << this->data2->get_p_data()[idx++] << std::endl;
      }

    }else{
      FAIL();
    }
    expected_output.close();
  }
  */

  /*
  std::fstream expected_output("tests/output/conv_forward.txt", std::ios_base::in);
  if(TypeParam::FUNC == FUNC_NOFUNC){
    T output;
    int idx = 0;
    if (expected_output.is_open()) {

      while (expected_output >> output)
        EXPECT_NEAR(this->data2->get_p_data()[idx++], output, EPS);

    }else{
      FAIL();
    }
    expected_output.close();
  }
  */
}

/*
TYPED_TEST(ParallelizedConvolutionBridgeTest, TestBackward){
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
      model >> this->ParallelizedConvolutionBridge_->p_model_cube->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  model.close();

  std::fstream bias_file("tests/input/conv_bias_in.txt", std::ios_base::in);
  if (bias_file.is_open()){
    for(int i=0;i<this->oD;i++){
      bias_file >> this->ParallelizedConvolutionBridge_->p_bias_cube->get_p_data()[i];
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


  this->ParallelizedConvolutionBridge_->forward();

  this->ParallelizedConvolutionBridge_->backward();
  //this->bias->logical_print();

  std::fstream expected_output("tests/output/conv_backward.txt", std::ios_base::in);
  T output;
  int idx = 0;

  if (expected_output.is_open()) {
    while (expected_output >> output)
      EXPECT_NEAR(this->grad1->get_p_data()[idx++], output, EPS);
  }else{
    FAIL();
  }
  expected_output.close();


  std::fstream expected_bias("tests/output/conv_bias.txt", std::ios_base::in);

  idx = 0;

  // TODO: I feel like this test is trivial -- please add some
  // non-trivial test case for bias term such that I can debug the bias term...
  if (expected_bias.is_open()) {
    while (expected_bias >> output) {
      float actual_bias = this->ParallelizedConvolutionBridge_->p_bias_cube->get_p_data()[idx];
      EXPECT_NEAR(actual_bias, output, EPS);
      idx++;
    }
  }else{
    FAIL();
  }
  expected_bias.close();

  std::fstream expected_weights("tests/output/conv_weights.txt", std::ios_base::in);

  idx = 0;

  if (expected_weights.is_open()) {
    while (expected_weights >> output) {
      float actual_weight = this->ParallelizedConvolutionBridge_->p_model_cube->get_p_data()[idx];
      EXPECT_NEAR(actual_weight, output, EPS);
      idx++;
    }
  }else{
    FAIL();
  }
  expected_weights.close();

}
*/
