#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/util.h"
#include "../src/Connector.h"
#include "../src/bridges/DropoutBridge.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

template <typename TypeParam>
class dropoutBridgeTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    dropoutBridgeTest() {
      data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad2 = new LogicalCube<T, Layout_CRDB> (iR, iC, iD, mB);

      layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
      layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

      cnn::LayerParameter layer_param;
      cnn::DropoutParameter * const dropout_param = layer_param.mutable_dropout_param();
      dropout_param->set_dropout_ratio(ratio);

      dropoutBridge_ = new DropoutBridge<T, Layout_CRDB, T, Layout_CRDB, CPUDriver>(layer1, layer2,
          &layer_param, &solver_param, &pdriver);
    }

    cnn::SolverParameter solver_param;
    CPUDriver pdriver;

    virtual ~dropoutBridgeTest() { delete layer1; delete layer2; }
    DropoutBridge<T, Layout_CRDB, T, Layout_CRDB, CPUDriver>* dropoutBridge_;

    LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;

    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    static const int mB = 4;
    static const int iD = 3;
    static const int iR = 6;
    static const int iC = 6;
    static constexpr float ratio = 0.5;
};

typedef ::testing::Types<FloatCRDB> DataTypes;

TYPED_TEST_CASE(dropoutBridgeTest, DataTypes);

//openblas_set_num_threads -- undefined reference -- currently disabled
TYPED_TEST(dropoutBridgeTest, TestInitialization) {
  EXPECT_TRUE(this->dropoutBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}

TYPED_TEST(dropoutBridgeTest, TestForward) {
  typedef typename TypeParam::T T;
  std::fstream input("tests/input/dropout_forward_in.txt", std::ios_base::in);
  if (input.is_open()){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
      input >> this->data1->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  input.close();

  std::fstream mask_cube_file("tests/input/dropout_mask.txt", std::ios_base::in);
  int m;
  int idx = 0;
  if (mask_cube_file.is_open()) {
    while (mask_cube_file >> m) {
      this->dropoutBridge_->mask_cube->get_p_data()[idx++] = m;
    }
  }else{
    FAIL();
  }
  mask_cube_file.close();

  this->dropoutBridge_->forward();

  std::fstream expected_output("tests/output/dropout_forward.txt", std::ios_base::in);

  T output;
  idx = 0;
  if (expected_output.is_open()) {
    while (expected_output >> output) {
      EXPECT_NEAR(this->data2->get_p_data()[idx++], output, EPS);
    }
  }else{
    FAIL();
  }
  expected_output.close();
}


TYPED_TEST(dropoutBridgeTest, TestBackward) {
  typedef typename TypeParam::T T;

  int oR = this->iR;
  int oC = this->iC;

  std::fstream grad("tests/input/dropout_backward_in.txt", std::ios_base::in);
  if (grad.is_open()){
    for(int i=0;i<oR*oC*this->iD*this->mB;i++){
      grad >> this->grad2->get_p_data()[i];
    }
  }
  else{
    FAIL();
  }
  grad.close();

  std::fstream mask_cube_file("tests/input/dropout_mask.txt", std::ios_base::in);
  int m;
  int idx = 0;
  if (mask_cube_file.is_open()) {
    while (mask_cube_file >> m) {
      this->dropoutBridge_->mask_cube->get_p_data()[idx++] = m;
    }
  }else{
    FAIL();
  }
  mask_cube_file.close();

  this->dropoutBridge_->forward();

  this->dropoutBridge_->backward();

  std::fstream expected_output("tests/output/dropout_backward", std::ios_base::in);

  T output;
  idx = 0;
  if (expected_output.is_open()) {
    while (expected_output >> output) {
      EXPECT_NEAR(this->grad1->get_p_data()[idx++], output, EPS);
    }
  }else{
    FAIL();
  }
  expected_output.close();
}

