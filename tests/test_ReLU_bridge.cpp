#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
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
class ReLUBridgeTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    ReLUBridgeTest() {
      data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad2 = new LogicalCube<T, Layout_CRDB> (iR, iC, iD, mB);

      layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
      layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

      cnn::LayerParameter layer_param;
      ReLUBridge_ = new ParallelizedBridge<T, ReLUBridge<T, Layout_CRDB, T, Layout_CRDB> >(layer1,
          layer2, &layer_param, &solver_param, 4, 1);
    }

    cnn::SolverParameter solver_param;

    virtual ~ReLUBridgeTest() { delete layer1; delete layer2; delete ReLUBridge_; }
    ParallelizedBridge<T, ReLUBridge<T, Layout_CRDB, T, Layout_CRDB> >* ReLUBridge_;

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

TYPED_TEST_CASE(ReLUBridgeTest, DataTypes);

//openblas_set_num_threads -- undefined reference -- currently disabled
TYPED_TEST(ReLUBridgeTest, TestInitialization) {
  EXPECT_TRUE(this->ReLUBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}

TYPED_TEST(ReLUBridgeTest, TestForward) {
  typedef typename TypeParam::T T;

  std::fstream input("tests/input/relu_forward_in.txt", std::ios_base::in);
  T * const data1 = this->data1->get_p_data();
  if (input.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
      input >> data1[i];
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
  const T * data2 = this->data2->get_p_data();
  if (expected_output.is_open()) {
    expected_output >> output;
    while (!expected_output.eof()) {
      EXPECT_NEAR(data2[idx], output, EPS);
      expected_output >> output;
      idx++;
    }
  }else{
    FAIL();
  }
  expected_output.close();
}


TYPED_TEST(ReLUBridgeTest, TestBackward) {
  typedef typename TypeParam::T T;

  T * const data1 = this->data1->get_p_data();
  std::fstream input("tests/input/relu_forward_in.txt", std::ios_base::in);
  if (input.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
      input >> data1[i];
    }
  }
  else{
    FAIL();
  }
  input.close();

  int oR = this->iR;
  int oC = this->iC;

  T * const data2 = this->data2->get_p_data();
  T * const grad2 = this->grad2->get_p_data();
  for(int i=0;i<oR*oC*this->iD*this->mB;i++) {
    data2[i] = 1;
    grad2[i] = i;
  }

  this->ReLUBridge_->forward();

  this->ReLUBridge_->backward();

  std::fstream expected_output("tests/output/relu_backward.txt", std::ios_base::in);

  T output;
  int idx = 0;
  const T * const grad1 = this->grad1->get_p_data();
  if (expected_output.is_open()) {
    expected_output >> output;
    while (!expected_output.eof()) {
      EXPECT_NEAR(grad1[idx], output, EPS);
      expected_output >> output;
      idx++;
    }
  }else{
    FAIL();
  }
  expected_output.close();
}

