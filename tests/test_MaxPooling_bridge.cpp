// Set default stride to 2 instead of 1
// No need to have a default kernel_size

#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/MaxPoolingBridge.h"
#include "../src/bridges/ParallelizedBridge.h"
#include "../src/util.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

template <typename TypeParam>
class MaxPoolingBridgeTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    MaxPoolingBridgeTest() {
      data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2 = new LogicalCube<T, Layout_CRDB>(oR, oC, iD, mB);
      grad2 = new LogicalCube<T, Layout_CRDB> (oR,oC, iD, mB);

      layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
      layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

      cnn::LayerParameter layer_param;
      cnn::PoolingParameter * const pool_param = layer_param.mutable_pooling_param();
      pool_param->set_kernel_size(k);
      pool_param->set_stride(s);

      MaxPoolingBridge_ = new ParallelizedBridge<DataType_SFFloat, MaxPoolingBridge<T, Layout_CRDB, T, Layout_CRDB> >(layer1,
          layer2, &layer_param, &solver_param, 4, 1);
    }

    virtual ~MaxPoolingBridgeTest() { delete layer1; delete layer2; }
    ParallelizedBridge<DataType_SFFloat, MaxPoolingBridge<T, Layout_CRDB, T, Layout_CRDB> >* MaxPoolingBridge_;

    cnn::SolverParameter solver_param;

    LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;

    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    static const int mB = 5;
    static const int iD = 3;
    static const int iR = 10;
    static const int iC = 10;
    static const int k = 4;
    static const int s = 2;
    static const int p = 0;
    static const int oR = static_cast<int>(0.5+(static_cast<float>(iR - k) / s)) + 1;
    static const int oC = static_cast<int>(0.5+(static_cast<float>(iC - k) / s)) + 1;
};

typedef ::testing::Types<FloatCRDB> DataTypes;

TYPED_TEST_CASE(MaxPoolingBridgeTest, DataTypes);

//openblas_set_num_threads -- undefined reference -- currently disabled
TYPED_TEST(MaxPoolingBridgeTest, TestInitialization) {
  EXPECT_TRUE(this->MaxPoolingBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}

TYPED_TEST(MaxPoolingBridgeTest, TestForward) {
  typedef typename TypeParam::T T;

  std::fstream input("tests/input/pooling_forward_in.txt", std::ios_base::in);
  if (input.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
      input >> this->data1->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  input.close();

  this->MaxPoolingBridge_->forward();

  std::fstream expected_output("tests/output/pooling_forward.txt", std::ios_base::in);

  T output;
  int idx = 0;
  if (expected_output.is_open()) {
    while (expected_output >> output) {
      EXPECT_NEAR(this->data2->get_p_data()[idx], output, EPS);
      idx++;
    }
  }
  else{
    FAIL();
  }
  expected_output.close();
}

TYPED_TEST(MaxPoolingBridgeTest, TestBackward) {
  typedef typename TypeParam::T T;

  std::fstream input("tests/input/pooling_forward_in.txt", std::ios_base::in);
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

  int oR = this->oR;
  int oC = this->oC;

  for(int i=0;i<oR*oC*this->iD*this->mB;i++) {
    this->grad2->get_p_data()[i] = i;
  }

  this->MaxPoolingBridge_->forward();

  this->MaxPoolingBridge_->backward();

  std::fstream expected_output("tests/output/pooling_backward.txt", std::ios_base::in);

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

