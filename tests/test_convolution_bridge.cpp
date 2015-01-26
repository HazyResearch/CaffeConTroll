//////// TO FIX -- Backward propogation giving wierd negative values
#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/ConvolutionBridge.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include "glog/logging.h"
#include <iostream>
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
    kernel1 = new LogicalCube<T, Layout_CRDB>(k, k, iD, oD);
    grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
    
    data2 = new LogicalCube<T, Layout_CRDB>(iR-k+1, iC-k+1, oD, mB);
    kernel2 = new LogicalCube<T, Layout_CRDB> (0, 0, iD, oD);
    grad2 = new LogicalCube<T, Layout_CRDB> (iR-k+1, iC-k+1, oD, mB);

    layer1 = new Layer<T, Layout_CRDB>(data1, kernel1, grad1);
    layer2 = new Layer<T, Layout_CRDB>(data2, kernel2, grad2);
    
    ConvolutionBridge_ = new ConvolutionBridge< CPU_CONV_LOWERINGTYPE1, TypeParam::FUNC, T, Layout_CRDB, T, Layout_CRDB>(layer1, layer2);
   } 

  	virtual ~ConvolutionBridgeTest() { delete ConvolutionBridge_; delete layer1; delete layer2;}
    ConvolutionBridge< CPU_CONV_LOWERINGTYPE1, TypeParam::FUNC, T, Layout_CRDB, T, Layout_CRDB>* ConvolutionBridge_;

  	LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* kernel1;
    LogicalCube<T, Layout_CRDB>* grad1;
    
    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* kernel2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    const int mB = 1;
    const int iD = 3;
    const int oD = 2;
    const int iR = 5;
    const int iC = 5;
    const int k = 3;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(ConvolutionBridgeTest, DataTypes);

//openblas_set_num_threads -- undefined reference -- currently disabled
TYPED_TEST(ConvolutionBridgeTest, TestInitialization){
  EXPECT_TRUE(this->ConvolutionBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}

TYPED_TEST(ConvolutionBridgeTest, TestForward){
	typedef typename TypeParam::T T;
	for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
        this->data1->p_data[i] = rand()%2;
    }
    for(int i=0;i<this->k*this->k*this->iD*this->oD;i++){
        this->kernel1->p_data[i] = rand()%2;
    }

    int oR = this->iR - this->k + 1;
    int oC = this->iC - this->k + 1;
    LogicalCube<T, Layout_CRDB>* out_expected = new LogicalCube<T, Layout_CRDB>(oR, oC, this->oD, this->mB);

    this->ConvolutionBridge_->forward();
    simple_conv<T>(this->data1, this->kernel1, out_expected);

    if(TypeParam::FUNC == FUNC_NOFUNC){
        for(int i=0; i<oR*oC*this->oD*this->mB;i++){
            EXPECT_NEAR(out_expected->p_data[i], this->data2->p_data[i],EPS);              
        }
    }
    else if(TypeParam::FUNC == FUNC_TANH){
        for(int i=0; i<oR*oC*this->oD*this->mB;i++){
            EXPECT_NEAR(tanh(out_expected->p_data[i]), this->data2->p_data[i],EPS);             
        }
    } 
}

// TODO ---- Test Backward

TYPED_TEST(ConvolutionBridgeTest, TestBackward){
    for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
        this->data1->p_data[i] = 1;
        this->grad1->p_data[i] = 1;
    }
    for(int i=0;i<this->k*this->k*this->iD*this->oD;i++){
        this->kernel1->p_data[i] = i;
    }

    int oR = this->iR - this->k + 1;
    int oC = this->iC - this->k + 1;

    for(int i=0;i<oR*oC*this->oD*this->mB;i++){
        this->data2->p_data[i] = 1;
        this->grad2->p_data[i] = i;
    }

    this->ConvolutionBridge_->forward();

    //std::cout << "\nGRADIENT=" << std::endl;
    //this->layer2->p_gradient_cube->logical_print();
    std::cout << "\nOLD WEIGHT=" << std::endl;
    this->layer1->p_model_cube->logical_print();

    
    //this->grad2->logical_print();
    this->ConvolutionBridge_->backward();
    
    std::cout << "\nNEW WEIGHT=" << std::endl;
    this->layer1->p_model_cube->logical_print();
    
    //std::cout << "\nNEW GRAD=" << std::endl;
    //this->layer1->p_gradient_cube->logical_print();
    //this->grad1->logical_print();
}

