#include "../src/Kernel.h"
#include "../src/Cube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/Bridge.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include "glog/logging.h"
#include <iostream>
#include <assert.h>
#include <cstring>

template <typename T>
void simple_conv(Cube<T, Layout_CRDB>* in, Cube<T, Layout_CRDB>* kernel, Cube<T, Layout_CRDB>* out){
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
class BridgeTest : public ::testing::Test {
 public:
  typedef typename TypeParam::T T;	
  BridgeTest(){
  	data1 = new Cube<T, Layout_CRDB>(5, 5, 1, 2);
    kernel1 = new Cube<T, Layout_CRDB>(3, 3, 1, 2);
    grad1 = new Cube<T, Layout_CRDB>(5, 5, 1, 2);
    
    data2 = new Cube<T, Layout_CRDB>(5-3+1, 5-3+1, 2, 2);
    kernel2 = new Cube<T, Layout_CRDB> (0, 0, 2, 2);
    grad2 = new Cube<T, Layout_CRDB> (5-3+1, 5-3+1, 2, 2);

    layer1 = new Layer<T, Layout_CRDB>(data1, kernel1, grad1);
    layer2 = new Layer<T, Layout_CRDB>(data2, kernel2, grad2);
    
    bridge_ = new Bridge<T, Layout_CRDB, T, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, TypeParam::FUNC>(layer1, layer2);
   } 

  	virtual ~BridgeTest() { delete bridge_; delete layer1; delete layer2;}
    Bridge<T, Layout_CRDB, T, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, TypeParam::FUNC>* bridge_;

  	Cube<T, Layout_CRDB>* data1;
    Cube<T, Layout_CRDB>* kernel1;
    Cube<T, Layout_CRDB>* grad1;
    
    Cube<T, Layout_CRDB>* data2;
    Cube<T, Layout_CRDB>* kernel2;
    Cube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(BridgeTest, DataTypes);

//openblas_set_num_threads -- undefined reference -- currently disabled
TYPED_TEST(BridgeTest, TestInitialization){
  EXPECT_TRUE(this->bridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}

TYPED_TEST(BridgeTest, TestForward){
	typedef typename TypeParam::T T;
	for(int i=0;i<5*5*2;i++){
        this->data1->p_data[i] = rand()%2;
    }
    for(int i=0;i<3*3*2;i++){
        this->kernel1->p_data[i] = rand()%2;
    }

    Cube<T, Layout_CRDB>* out_expected = new Cube<T, Layout_CRDB>(5-3+1, 5-3+1, 2, 2);

    this->bridge_->forward();
    simple_conv<T>(this->data1, this->kernel1, out_expected);
    std::cout << "\nINPUT DATA=" << std::endl;
    this->layer1->p_data_cube->logical_print();
    std::cout << "\nMODEL DATA=" << std::endl;
    this->layer1->p_model_cube->logical_print();
    std::cout << "\nOUTPUT DATA=" << std::endl;
    this->layer2->p_data_cube->logical_print();
    std::cout << "\nEXPECTED OUTPUT DATA=" << std::endl;
    out_expected->logical_print();
}

/*
TYPED_TEST(BridgeTest, TestBackward){
	for(int i=0;i<5*5*2;i++){
        this->data1->p_data[i] = rand()%2;
        this->grad1->p_data[i] = rand()%2;
    }
    for(int i=0;i<3*3*3;i++){
        this->kernel1->p_data[i] = rand()%2;
    }
    for(int i=0;i<3*3*3*2;i++){
        this->data2->p_data[i] = rand()%2;
        this->grad2->p_data[i] = rand()%2;
    }
    this->bridge_->forward();
    std::cout << "\nGRADIENT=" << std::endl;
    this->layer2->p_gradient_cube->logical_print();

    std::cout << "\nOLD WEIGHT=" << std::endl;
    this->layer1->p_model_cube->logical_print();
    
    this->bridge_->backward();
    
    std::cout << "\nNEW WEIGHT=" << std::endl;
    this->layer1->p_model_cube->logical_print();
    
    std::cout << "\nNEW GRAD=" << std::endl;
    this->layer1->p_gradient_cube->logical_print();
}
*/
