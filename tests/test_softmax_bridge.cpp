#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/SoftmaxLossBridge.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

template <typename TypeParam>
class softmaxBridgeTest : public ::testing::Test {
 public:
  typedef typename TypeParam::T T;	
  softmaxBridgeTest(){
  	data1 = new LogicalCube<T, Layout_CRDB>(1, 1, iD, mB);
    grad1 = new LogicalCube<T, Layout_CRDB>(1, 1, iD, mB);
    
    data2 = new LogicalCube<T, Layout_CRDB>(1, 1, iD, mB);
    grad2 = new LogicalCube<T, Layout_CRDB> (1, 1, iD, mB);

    label = new LogicalCube<T, Layout_CRDB> (1, 1, 1, mB);

    layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
    layer2 = new Layer<T, Layout_CRDB>(data2, grad2);
    
    softmaxBridge_ = new SoftmaxLossBridge<T, Layout_CRDB, T, Layout_CRDB>(layer1, layer2, label);
   } 

  	virtual ~softmaxBridgeTest() { delete softmaxBridge_; delete layer1; delete layer2;}
    SoftmaxLossBridge<T, Layout_CRDB, T, Layout_CRDB>* softmaxBridge_;

  	LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;
    
    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    LogicalCube<T, Layout_CRDB>* label;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    static const int mB = 5;
    static const int iD = 100;
};
    
typedef ::testing::Types<FloatCRDB> DataTypes;

TYPED_TEST_CASE(softmaxBridgeTest, DataTypes);

//openblas_set_num_threads -- undefined reference -- currently disabled
TYPED_TEST(softmaxBridgeTest, TestInitialization){
  EXPECT_TRUE(this->softmaxBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
  EXPECT_TRUE(this->label);
}

TYPED_TEST(softmaxBridgeTest, TestForward){
	typedef typename TypeParam::T T;

    srand(1);  
    for(int i=0;i<this->iD*this->mB;i++){
        this->data1->p_data[i] = (rand()%5)*0.1;
    }

    srand(0);
    for(int n=0;n<this->mB;n++){
        this->label->p_data[n] = rand()%10;
    }
    this->softmaxBridge_->forward();
    std::fstream expected_output("softmax_forward.txt", std::ios_base::in);
    
    T output;
    int idx = 0;
    if (expected_output.is_open()) {
        expected_output >> output;
        EXPECT_NEAR(this->softmaxBridge_->loss, output, EPS);
        expected_output >> output;
    }
    expected_output.close();
}


TYPED_TEST(softmaxBridgeTest, TestBackward){
    typedef typename TypeParam::T T;
    
    srand(0);
    for(int n=0;n<this->mB;n++){
        this->label->p_data[n] = rand()%10;
    }


    this->softmaxBridge_->forward();
    srand(1);
    for(int i=0;i<this->iD*this->mB;i++){
        this->data2->p_data[i] = (rand()%5)*0.1;
    }
    this->softmaxBridge_->backward();

    std::fstream expected_output("softmax_backward.txt", std::ios_base::in);
    
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
}

