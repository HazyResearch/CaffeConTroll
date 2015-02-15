#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
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

      dropoutBridge_ = new DropoutBridge<T, Layout_CRDB, T, Layout_CRDB>(layer1, layer2, &layer_param, &solver_param);
    }

    cnn::SolverParameter solver_param;


    virtual ~dropoutBridgeTest() { delete dropoutBridge_; delete layer1; delete layer2;}
    DropoutBridge<T, Layout_CRDB, T, Layout_CRDB>* dropoutBridge_;

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
  srand(1);
  typedef typename TypeParam::T T;
  for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++) {
    this->data1->p_data[i] = rand()%10;
  }

  std::fstream mask_cube_file("dropout_mask.txt", std::ios_base::in);
  int m;
  int idx = 0;
  if (mask_cube_file.is_open()) {
    mask_cube_file >> m;
    while (!mask_cube_file.eof()) {
      this->dropoutBridge_->mask_cube->p_data[idx] = m;
      mask_cube_file >> m;
      idx++;
    }
  }
  mask_cube_file.close();

  this->dropoutBridge_->forward();

  std::fstream expected_output("dropout_forward.txt", std::ios_base::in);

  T output;
  idx = 0;
  if (expected_output.is_open()) {
    expected_output >> output;
    while (!expected_output.eof()) {
      EXPECT_NEAR(this->data2->p_data[idx], output, EPS);
      expected_output >> output;
      idx++;
    }
  }
  expected_output.close();
}


TYPED_TEST(dropoutBridgeTest, TestBackward) {
  typedef typename TypeParam::T T;


  int oR = this->iR;
  int oC = this->iC;

  srand(0);
  for(int i=0;i<oR*oC*this->iD*this->mB;i++) {
    this->grad2->p_data[i] = rand()%10;
  }

  std::fstream mask_cube_file("dropout_mask_cube.txt", std::ios_base::in);
  int m;
  int idx = 0;
  if (mask_cube_file.is_open()) {
    mask_cube_file >> m;
    while (!mask_cube_file.eof()) {
      this->dropoutBridge_->mask_cube->p_data[idx] = m;
      mask_cube_file >> m;
      idx++;
    }
  }
  mask_cube_file.close();

  this->dropoutBridge_->forward();

  this->dropoutBridge_->backward();

  std::fstream expected_output("dropout_backward.txt", std::ios_base::in);

  T output;
  idx = 0;
  if (expected_output.is_open()) {
    expected_output >> output;
    while (!expected_output.eof()) {
      EXPECT_NEAR(this->grad1->p_data[idx], output, EPS);
      expected_output >> output;
      idx++;
    }
  }
  expected_output.close();
}

