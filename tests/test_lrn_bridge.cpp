#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/LRNBridge.h"
#include "../src/bridges/ParallelizedBridge.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

// const double ESP2 = 0.001;  // We increase this threshold becasue
                            // of the fastPrecisePow function that we are using.

template <typename TypeParam>
class LRNBridgeTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    LRNBridgeTest(){
      data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad2 = new LogicalCube<T, Layout_CRDB> (iR, iC, iD, mB);

      layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
      layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

      cnn::LayerParameter layer_param;
      cnn::LRNParameter * const lrn_param = layer_param.mutable_lrn_param();
      lrn_param->set_alpha(alpha);
      lrn_param->set_beta(beta);
      lrn_param->set_local_size(local_size);

      LRNBridge_ = new ParallelizedBridge<DataType_SFFloat, LRNBridge<T, Layout_CRDB, T, Layout_CRDB> >(layer1, layer2,
          &layer_param, &solver_param, 4, 1);
    }

    cnn::SolverParameter solver_param;

    virtual ~LRNBridgeTest() { delete data1; delete data2; delete grad1; delete grad2; delete layer1; delete layer2;}
    ParallelizedBridge<DataType_SFFloat, LRNBridge<T, Layout_CRDB, T, Layout_CRDB> >* LRNBridge_;

    LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;

    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    static const int mB = 6;
    static const int iD = 3;
    static const int iR = 10;
    static const int iC = 10;
    static constexpr float alpha = 0.0001;
    static constexpr float beta = 0.75;
    static const int local_size = 3;
};

typedef ::testing::Types<FloatCRDB> DataTypes;

TYPED_TEST_CASE(LRNBridgeTest, DataTypes);

//openblas_set_num_threads -- undefined reference -- currently disabled
TYPED_TEST(LRNBridgeTest, TestInitialization){
  EXPECT_TRUE(this->LRNBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}

TYPED_TEST(LRNBridgeTest, TestForward){
  typedef typename TypeParam::T T;

  std::fstream input("tests/input/lrn_forward_in.txt", std::ios_base::in);
  if (input.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
      input >> this->data1->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  input.close();

  this->LRNBridge_->forward();

  std::fstream expected_output("tests/output/lrn_forward.txt", std::ios_base::in);

  T output;
  int idx = 0;
  if (expected_output.is_open()) {
    while (expected_output >> output) {
      EXPECT_NEAR(this->data2->get_p_data()[idx++], output, EPS);
    }
  }else{
    FAIL();
  }
  expected_output.close();
}


TYPED_TEST(LRNBridgeTest, TestBackward){
  typedef typename TypeParam::T T;

  std::fstream input("tests/input/lrn_forward_in.txt", std::ios_base::in);
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

  this->LRNBridge_->forward();

  std::fstream grad("tests/input/lrn_backward_in.txt", std::ios_base::in);
  if (grad.is_open()){
    for(int i=0;i<oR*oC*this->iD*this->mB;i++){
      grad >> this->grad2->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  grad.close();

  this->LRNBridge_->backward();

  std::fstream expected_output("tests/output/lrn_backward.txt", std::ios_base::in);
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
}

