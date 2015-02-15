#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/ConvolutionBridge.h"
#include "../src/bridges/FullyConnectedBridge.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>


template <typename TypeParam>
class FCBridgeTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    FCBridgeTest() {
      data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2 = new LogicalCube<T, Layout_CRDB>(oR, oC, oD, mB);
      grad2 = new LogicalCube<T, Layout_CRDB> (oR, oC, oD, mB);

      layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
      layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

      data1c = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1c = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2c = new LogicalCube<T, Layout_CRDB>(oR, oC, oD, mB);
      grad2c = new LogicalCube<T, Layout_CRDB> (oR, oC, oD, mB);

      layer1c = new Layer<T, Layout_CRDB>(data1c, grad1c);
      layer2c = new Layer<T, Layout_CRDB>(data2c, grad2c);

      cnn::SolverParameter solver_param;

      cnn::LayerParameter layer_param;
      cnn::ConvolutionParameter * const conv_param = layer_param.mutable_convolution_param();
      conv_param->set_num_output(oD);
      conv_param->set_kernel_size(k);
      conv_param->set_pad(p);
      conv_param->set_stride(s);

      cnn::InnerProductParameter * const inn_param = layer_param.mutable_inner_product_param();
      inn_param->set_num_output(oD);

      ConvolutionBridge_ = new ConvolutionBridge< CPU_CONV_LOWERINGTYPE1, TypeParam::FUNC, T, Layout_CRDB, T, Layout_CRDB>(layer1,
          layer2, &layer_param, &solver_param);

      FullyConnectedBridge_ = new FullyConnectedBridge< T, Layout_CRDB, T, Layout_CRDB>(layer1c,
          layer2c, &layer_param, &solver_param);
    }

    ConvolutionBridge< CPU_CONV_LOWERINGTYPE1, TypeParam::FUNC, T, Layout_CRDB, T, Layout_CRDB>* ConvolutionBridge_;
    FullyConnectedBridge< T, Layout_CRDB, T, Layout_CRDB>* FullyConnectedBridge_;

    LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;

    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    LogicalCube<T, Layout_CRDB>* data1c;
    LogicalCube<T, Layout_CRDB>* grad1c;

    LogicalCube<T, Layout_CRDB>* data2c;
    LogicalCube<T, Layout_CRDB>* grad2c;

    Layer<T, Layout_CRDB>* layer1c;
    Layer<T, Layout_CRDB>* layer2c;

    static const int mB = 7;
    static const int iD = 12;
    static const int oD = 25;
    static const int iR = 20;
    static const int iC = 20;
    static const int k = 20;
    static const int s = 1;
    static const int p = 0;
    static const int oR = static_cast<int>((static_cast<float>(iR + 2*p - k) / s)) + 1;
    static const int oC = static_cast<int>((static_cast<float>(iC + 2*p - k) / s)) + 1;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(FCBridgeTest, DataTypes);

TYPED_TEST(FCBridgeTest, TestInitialization) {
  EXPECT_TRUE(this->ConvolutionBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
  EXPECT_TRUE(this->FullyConnectedBridge_);
  EXPECT_TRUE(this->layer1c);
  EXPECT_TRUE(this->layer2c);
}

TYPED_TEST(FCBridgeTest, TestForward) {
  typedef typename TypeParam::T T;
  srand(1);
  for (int i=0;i<this->iR*this->iC*this->iD*this->mB;i++) {
    this->data1->p_data[i] = rand()%10;
    this->data1c->p_data[i] = this->data1->p_data[i];
  }
  srand(0);
  for (int i=0;i<this->k*this->k*this->iD*this->oD;i++) {
    this->ConvolutionBridge_->get_model_cube()->p_data[i] = rand()%10;
    this->FullyConnectedBridge_->get_model_cube()->p_data[i] = 
      this->ConvolutionBridge_->get_model_cube()->p_data[i];
  }
  srand(0);
  for (int i=0;i<this->oD;i++) {
    this->ConvolutionBridge_->get_bias_cube()->p_data[i] = 0.1*(rand()%10);
    this->FullyConnectedBridge_->get_bias_cube()->p_data[i] = 
      this->ConvolutionBridge_->get_bias_cube()->p_data[i];
  }

  int oR = this->oR;
  int oC = this->oC;

  this->ConvolutionBridge_->forward();

  this->FullyConnectedBridge_->forward();

  for (int i=0;i<this->oR*this->oC*this->oD*this->mB;i++) {
    EXPECT_NEAR(this->data2c->p_data[i], this->data2->p_data[i], EPS);
  }

}


TYPED_TEST(FCBridgeTest, TestBackward) {
  typedef typename TypeParam::T T;
  srand(1);
  for (int i=0;i<this->iR*this->iC*this->iD*this->mB;i++) {
    this->data1->p_data[i] = rand()%10;
    this->data1c->p_data[i] = this->data1->p_data[i];
    this->grad1->p_data[i] = 0;
    this->grad1c->p_data[i] = this->grad1->p_data[i];
  }

  srand(0);
  for (int i=0;i<this->k*this->k*this->iD*this->oD;i++) {
    this->ConvolutionBridge_->get_model_cube()->p_data[i] = rand()%2;
    this->FullyConnectedBridge_->get_model_cube()->p_data[i] = 
      this->ConvolutionBridge_->get_model_cube()->p_data[i];
  }

  int oR = this->oR;
  int oC = this->oC;

  for (int i=0;i<oR*oC*this->oD*this->mB;i++) {
    this->data2->p_data[i] = 0;
    this->grad2->p_data[i] = i*0.1;
    this->data2c->p_data[i] = 0;
    this->grad2c->p_data[i] = i*0.1;
  }

  srand(0);
  for (int i=0;i<this->oD;i++) {
    this->ConvolutionBridge_->get_bias_cube()->p_data[i] = 0.1*(rand()%10);
    this->FullyConnectedBridge_->get_bias_cube()->p_data[i] =
      this->ConvolutionBridge_->get_bias_cube()->p_data[i];
  }

  this->ConvolutionBridge_->forward();
  this->FullyConnectedBridge_->forward();

  this->ConvolutionBridge_->backward();
  this->FullyConnectedBridge_->backward();


  for (int i=0;i<this->iR*this->iC*this->iD*this->mB;i++) {
    EXPECT_NEAR(this->grad1->p_data[i], this->grad1c->p_data[i], EPS);
  }
  
  for (int i=0;i<this->oD;i++) {
    EXPECT_NEAR(this->FullyConnectedBridge_->get_bias_cube()->p_data[i], 
      this->ConvolutionBridge_->get_bias_cube()->p_data[i], EPS);
  }

  for (int i=0;i<this->k*this->k*this->iD*this->oD;i++) {
    EXPECT_NEAR(this->FullyConnectedBridge_->get_model_cube()->p_data[i], 
      this->ConvolutionBridge_->get_model_cube()->p_data[i], EPS);
  }

}
