#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/ConvolutionBridge.h"
#include "test_types.h"
#include "gtest/gtest.h"
// #include "glog/logging.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

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
class ConvolutionBridgeTest : public ::testing::Test {
 public:
  typedef typename TypeParam::T T;	
  ConvolutionBridgeTest(){
  	data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
    grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
    
    data2 = new LogicalCube<T, Layout_CRDB>(oR, oC, oD, mB);
    grad2 = new LogicalCube<T, Layout_CRDB> (oR, oC, oD, mB);

    // kernel = new LogicalCube<T, Layout_CRDB> (k, k, iD, oD);
    // bias = new LogicalCube<T, Layout_CRDB> (1, 1, oD, 1);

    layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
    layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

    bconfig = new BridgeConfig(k, oD, p, s, weight_initializer, bias_initializer);

    ConvolutionBridge_ = new ConvolutionBridge< CPU_CONV_LOWERINGTYPE1, TypeParam::FUNC, T, Layout_CRDB, T, Layout_CRDB>(layer1, layer2, bconfig);
   } 

  	virtual ~ConvolutionBridgeTest() { delete ConvolutionBridge_; delete layer1; delete layer2; delete bconfig;}
    ConvolutionBridge< CPU_CONV_LOWERINGTYPE1, TypeParam::FUNC, T, Layout_CRDB, T, Layout_CRDB>* ConvolutionBridge_;

  	LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;
    
    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    // LogicalCube<T, Layout_CRDB>* kernel;
    // LogicalCube<T, Layout_CRDB>* bias;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    BridgeConfig * bconfig;

    static const int mB = 4;
    static const int iD = 3;
    static const int oD = 10;
    static const int iR = 20;
    static const int iC = 20;
    static const int k = 5;
    static const int s = 4;
    static const int p = 2;
    static const int oR = static_cast<int>(floor(static_cast<float>(iR + 2*p - k) / s)) + 1;
    static const int oC = static_cast<int>(floor(static_cast<float>(iC + 2*p - k) / s)) + 1;
    static const InitializerType weight_initializer = XAVIER;
    static const InitializerType bias_initializer = CONSTANT;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(ConvolutionBridgeTest, DataTypes);

TYPED_TEST(ConvolutionBridgeTest, TestInitialization){
  EXPECT_TRUE(this->ConvolutionBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}

TYPED_TEST(ConvolutionBridgeTest, TestForward){
	typedef typename TypeParam::T T;
    srand(1);
	for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
        this->data1->p_data[i] = rand()%10;
    }
    srand(0);
    for(int i=0;i<this->k*this->k*this->iD*this->oD;i++){
        this->ConvolutionBridge_->model_cube()->p_data[i] = rand()%10;
    }
    srand(0);
    for(int i=0;i<this->oD;i++){
        this->ConvolutionBridge_->bias_cube()->p_data[i] = 0.1*(rand()%10);
    }

    int oR = this->oR;
    int oC = this->oC;
    
    this->ConvolutionBridge_->forward();
    
    std::fstream expected_output("conv_forward.txt", std::ios_base::in);
    if(TypeParam::FUNC == FUNC_NOFUNC){
        T output;
        int idx = 0;
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
}


TYPED_TEST(ConvolutionBridgeTest, TestBackward){
    typedef typename TypeParam::T T;
    srand(1);
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
        this->data1->p_data[i] = rand()%10;
        this->grad1->p_data[i] = 0;
    }
    
    srand(0);
    for(int i=0;i<this->k*this->k*this->iD*this->oD;i++){
        this->ConvolutionBridge_->model_cube()->p_data[i] = rand()%2;
    }
    
    int oR = this->oR;
    int oC = this->oC;

    for(int i=0;i<oR*oC*this->oD*this->mB;i++){
        this->data2->p_data[i] = 0;
        this->grad2->p_data[i] = i*0.1;
    }

    for(int i=0;i<this->oD;i++){
        this->ConvolutionBridge_->bias_cube()->p_data[i] = 0.0;
    }

    this->ConvolutionBridge_->forward();

    this->ConvolutionBridge_->backward();
    //this->bias->logical_print();
    std::fstream expected_output("conv_backward.txt", std::ios_base::in);
    T output;
    int idx = 0;
    if (expected_output.is_open()) {
        expected_output >> output;
        while (!expected_output.eof()) {
            EXPECT_NEAR(this->grad1->p_data[idx], output, EPS);
            expected_output >> output;
            idx++;
        }
    }
    expected_output.close(); 

    std::fstream expected_weights("conv_weights.txt", std::ios_base::in);
    idx = 0;
    if (expected_weights.is_open()) {
        expected_weights >> output;
        while (!expected_weights.eof()) {
            EXPECT_NEAR(this->ConvolutionBridge_->model_cube()->p_data[idx], output, 0.9);
            expected_weights >> output;
            idx++;
        }
    }
    expected_weights.close(); 

}

