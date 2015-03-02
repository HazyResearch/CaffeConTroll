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

  	virtual ~softmaxBridgeTest() { delete data1; delete data2; delete grad1; delete grad2; delete layer1; delete layer2;}
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

    std::fstream input("tests/input/softmax_forward_in.txt", std::ios_base::in);
    if (input.is_open()){
        for(int i=0;i<this->iD*this->mB;i++){
          input >> this->data1->get_p_data()[i];
        }
    }
    else{
        FAIL();
    }
    input.close();

    std::fstream label_file("tests/input/softmax_label.txt", std::ios_base::in);
    if (label_file.is_open()){
        for(int i=0;i<this->mB;i++){
          label_file >> this->label->get_p_data()[i];
        }
    }
    else{
        FAIL();
    }
    label_file.close();

    this->softmaxBridge_->forward();
    std::fstream expected_output("tests/output/softmax_forward.txt", std::ios_base::in);

    T output;
    if (expected_output.is_open()) {
        expected_output >> output;
        EXPECT_NEAR(this->softmaxBridge_->get_loss(), output, EPS);
        expected_output >> output;
    }else{
        FAIL();
    }
    expected_output.close();
}


TYPED_TEST(softmaxBridgeTest, TestBackward){
    typedef typename TypeParam::T T;

    std::fstream label_file("tests/input/softmax_label.txt", std::ios_base::in);
    if (label_file.is_open()){
        for(int i=0;i<this->mB;i++){
          label_file >> this->label->get_p_data()[i];
        }
    }
    else{
        FAIL();
    }
    label_file.close();

    this->softmaxBridge_->forward();

    std::fstream output_file("tests/input/softmax_backward_in.txt", std::ios_base::in);
    if (output_file.is_open()){
        for(int i=0;i<this->iD*this->mB;i++){
          output_file >> this->data2->get_p_data()[i];
        }
    }
    else{
        FAIL();
    }
    output_file.close();

    this->softmaxBridge_->backward();

    std::fstream expected_output("tests/output/softmax_backward.txt", std::ios_base::in);

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

