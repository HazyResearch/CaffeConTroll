#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/util.h"
#include "../src/Connector.h"
#include "../src/bridges/ConvolutionBridge.h"
#include "../src/bridges/ParallelizedBridge.h"
#include "../src/sched/DeviceDriver_GPU.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

#define WRITE_MODE 0

template <typename TypeParam>
class ParallelizedConvolutionBridgeLargeCPU_GPU_batchTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    ParallelizedConvolutionBridgeLargeCPU_GPU_batchTest(){
      data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2 = new LogicalCube<T, Layout_CRDB>(oR, oC, oD, mB);
      grad2 = new LogicalCube<T, Layout_CRDB> (oR, oC, oD, mB);

      layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
      layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

      cnn::LayerParameter layer_param;
      layer_param.set_gpu_0_batch_proportion(0.5);
      cnn::ConvolutionParameter * const conv_param = layer_param.mutable_convolution_param();
      conv_param->set_num_output(oD);
      conv_param->set_kernel_size(k);
      conv_param->set_pad(p);
      conv_param->set_stride(s);

      solver_param.set_base_lr(0.01);
      solver_param.set_momentum(0.0);
      solver_param.set_lr_policy("step");
      solver_param.set_stepsize(10000);


      // TODO: set #partition to 8 does not halt
      ParallelizedConvolutionBridge_ = new ParallelizedBridge<DataType_SFFloat,
              ConvolutionBridge>(layer1,
                  layer2, &layer_param, &solver_param, pdriver, 4, 1);

      ParallelizedConvolutionBridge_->needs_to_calc_backward_grad = true;
    }

    virtual ~ParallelizedConvolutionBridgeLargeCPU_GPU_batchTest() { 
        delete layer1; 
        delete layer2; 
        delete ParallelizedConvolutionBridge_;
    }
    
    ParallelizedBridge<DataType_SFFloat,
              ConvolutionBridge>* ParallelizedConvolutionBridge_;

    LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;

    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    cnn::SolverParameter solver_param;

    CPUDriver * const pdriver = new CPUDriver();

    static const int mB = 4;
    static const int iD = 3;
    static const int oD = 8;
    static const int iR = 127;
    static const int iC = 127;
    static const int k = 11;
    static const int s = 4;
    static const int p = 2;
    static const int oR = static_cast<int>((static_cast<float>(iR + 2*p - k) / s)) + 1;
    static const int oC = static_cast<int>((static_cast<float>(iC + 2*p - k) / s)) + 1;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(ParallelizedConvolutionBridgeLargeCPU_GPU_batchTest, DataTypes);

TYPED_TEST(ParallelizedConvolutionBridgeLargeCPU_GPU_batchTest, TestInitialization){
  EXPECT_TRUE(this->ParallelizedConvolutionBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}


TYPED_TEST(ParallelizedConvolutionBridgeLargeCPU_GPU_batchTest, TestForward){
  typedef typename TypeParam::T T;

  std::fstream input("tests/input/conv_forward_in_large.txt", std::ios_base::in); // File size: iR*iC*iD*mB = 127*127*3*4
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

  std::fstream model("tests/input/conv_model_large.txt", std::ios_base::in); // File size: k*k*iD*oD = 11*11*3*8
  if (model.is_open()){
    for(int i=0;i<this->k*this->k*this->iD*this->oD;i++){
      model >> this->ParallelizedConvolutionBridge_->p_model_cube->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  model.close();

  std::fstream bias_file("tests/input/conv_bias_in_large.txt", std::ios_base::in); // File size: oD = 8
  if (bias_file.is_open()){
    for(int i=0;i<this->oD;i++){
      bias_file >> this->ParallelizedConvolutionBridge_->p_bias_cube->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  bias_file.close();

  this->ParallelizedConvolutionBridge_->forward();

#if WRITE_MODE
  std::ofstream expected_output("tests/output/conv_forward_large.txt"); // File size: oD*m*m*mB
  if (expected_output.is_open()) {
    for (int i=0; i<this->oC*this->oR*this->mB*this->oD; ++i) {
      expected_output << this->data2->get_p_data()[i] << " ";
    }
  }
#else
  std::fstream expected_output("tests/output/conv_forward_large.txt", std::ios_base::in); // File size: oD*m*m*b = 
  if(TypeParam::FUNC == FUNC_NOFUNC){
    T output;
    int idx = 0;
    if (expected_output.is_open()) {

      while (expected_output >> output)
        EXPECT_NEAR(this->data2->get_p_data()[idx++], output, std::max(0.01,std::abs(output/100.0)));

    }else{
      FAIL();
    }
    expected_output.close();
  }
#endif
}

TYPED_TEST(ParallelizedConvolutionBridgeLargeCPU_GPU_batchTest, TestBackward){
  typedef typename TypeParam::T T;

  std::fstream input("tests/input/conv_forward_in_large.txt", std::ios_base::in);
  if (input.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
      input >> this->data1->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  input.close();

  std::fstream model("tests/input/conv_backward_model_large.txt", std::ios_base::in); // File size: k*k*iD*oD
  if (model.is_open()){
    for(int i=0;i<this->k*this->k*this->iD*this->oD;i++){
      model >> this->ParallelizedConvolutionBridge_->p_model_cube->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  model.close();

  std::fstream bias_file("tests/input/conv_bias_in_large.txt", std::ios_base::in);
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

#if WRITE_MODE
  std::ofstream expected_output("tests/output/conv_backward_large.txt"); // File size: oD*m*m*mB
  if (expected_output.is_open()) {
    for (int i=0; i<this->oC*this->oR*this->mB*this->oD; ++i) {
      expected_output << this->grad1->get_p_data()[i] << " ";
    }
  }
  expected_output.close();
#else
  std::fstream expected_output("tests/output/conv_backward_large.txt", std::ios_base::in); // File size: oD*m*m*mB
  T output;
  int idx = 0;

  if (expected_output.is_open()) {
    while (expected_output >> output)
      EXPECT_NEAR(this->grad1->get_p_data()[idx++], output, std::max(0.01,std::abs(output/100.0)));
  }else{
    FAIL();
  }
  expected_output.close();
#endif

#if WRITE_MODE
  std::ofstream expected_bias("tests/output/conv_bias_large.txt");
  if (expected_bias.is_open()) {
    for (int i=0; i<this->oD; ++i) {
      expected_bias << this->ParallelizedConvolutionBridge_->p_bias_cube->get_p_data()[i] << " ";
    }
  }
  expected_bias.close();
#else
  std::fstream expected_bias("tests/output/conv_bias_large.txt", std::ios_base::in); // File size: oD = 8
  idx = 0;
  if (expected_bias.is_open()) {
    while (expected_bias >> output) {
      float actual_bias = this->ParallelizedConvolutionBridge_->p_bias_cube->get_p_data()[idx];
      EXPECT_NEAR(actual_bias, output, std::max(0.01,std::abs(output/100.0)));
      idx++;
    }
  }else{
    FAIL();
  }
  expected_bias.close();
#endif

#if WRITE_MODE
  std::ofstream expected_weights("tests/output/conv_weights_large.txt");
  if (expected_weights.is_open()) {
    for (int i=0; i<this->k*this->k*this->oD*this->iD; ++i) {
      expected_weights << this->ParallelizedConvolutionBridge_->p_model_cube->get_p_data()[i] << " ";
    }
  }
  expected_weights.close();
#else
  std::fstream expected_weights("tests/output/conv_weights_large.txt", std::ios_base::in); // k*k*oD*iD = 750
  idx = 0;
  if (expected_weights.is_open()) {
    while (expected_weights >> output) {
      float actual_weight = this->ParallelizedConvolutionBridge_->p_model_cube->get_p_data()[idx];
      EXPECT_NEAR(actual_weight, output, std::max(0.01,std::abs(output/100.0)));
      idx++;
    }
  }else{
    FAIL();
  }
  expected_weights.close();
#endif

}

