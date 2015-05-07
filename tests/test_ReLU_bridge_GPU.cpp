#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/util.h"
#include "../src/Connector.h"
#include "../src/bridges/ReLUBridge.h"
#include "../src/bridges/ParallelizedBridge.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

template <typename TypeParam>
class GPUReLUBridgeTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    GPUReLUBridgeTest() {
      data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad2 = new LogicalCube<T, Layout_CRDB> (iR, iC, iD, mB);

      layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
      layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

      cnn::LayerParameter layer_param;
      layer_param.set_gpu_0_batch_proportion(1);
      ReLUBridge_ = new ParallelizedBridge<T, ReLUBridge>(layer1, layer2, &layer_param, &solver_param, &pdriver, 4, 1);
    }

    cnn::SolverParameter solver_param;
    CPUDriver pdriver;

    virtual ~GPUReLUBridgeTest() { delete layer1; delete layer2; delete ReLUBridge_; }
    ParallelizedBridge<T, ReLUBridge>* ReLUBridge_;

    LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;

    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    static const int mB = 5;
    static const int iD = 3;
    static const int iR = 20;
    static const int iC = 20;
};

typedef ::testing::Types<FloatCRDB> DataTypes;

TYPED_TEST_CASE(GPUReLUBridgeTest, DataTypes);

//openblas_set_num_threads -- undefined reference -- currently disabled
TYPED_TEST(GPUReLUBridgeTest, TestInitialization) {
  EXPECT_TRUE(this->ReLUBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}

TYPED_TEST(GPUReLUBridgeTest, TestForward) {
  typedef typename TypeParam::T T;

  std::fstream input("tests/input/relu_forward_in.txt", std::ios_base::in);
  if (input.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
      input >> this->data1->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  input.close();

  this->ReLUBridge_->forward();

  std::fstream expected_output("tests/output/relu_forward.txt", std::ios_base::in);

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


TYPED_TEST(GPUReLUBridgeTest, TestBackward) {
  typedef typename TypeParam::T T;

  std::fstream input("tests/input/relu_forward_in.txt", std::ios_base::in);
  if (input.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
      input >> this->data1->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  input.close();

  int oR = this->iR;
  int oC = this->iC;

  for(int i=0;i<oR*oC*this->iD*this->mB;i++) {
    this->data2->get_p_data()[i] = 1;
    this->grad2->get_p_data()[i] = i;
  }

  this->ReLUBridge_->forward();

  this->ReLUBridge_->backward();

  std::fstream expected_output("tests/output/relu_backward.txt", std::ios_base::in);

  T output;
  int idx = 0;
  if (expected_output.is_open()) {
    while (expected_output >> output)
      EXPECT_NEAR(this->grad1->get_p_data()[idx++], output, EPS);
  }else{
    FAIL();
  }
  expected_output.close();
}

