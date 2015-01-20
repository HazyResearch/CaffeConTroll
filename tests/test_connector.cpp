#include "../src/Connector.h"
#include "../src/Cube.h"
#include "test_types.h"
#include <cstring>
#include "gtest/gtest.h"
#include "glog/logging.h"
#include <iostream>
#include <assert.h>
using namespace std;

template <typename TypeParam>
class ConnectorTest : public ::testing::Test {
 protected:
 	typedef typename TypeParam::T T;
  ConnectorTest(){
  	input_cube = new Cube<T, TypeParam::LAYOUT>(4,4,2,2);
 	p_config = new LoweringConfig();
	p_config->kernel_size = 3;

	output_cube = new Cube<T, Layout_CRDB>(18,8,1,1);
  	connector_ = new Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, Connector_Lowering_TYPE1>(input_cube, output_cube, 
		p_config);	
  }
      //: connector_(new Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, Connector_Lowering_Type1>(4, 5, 3, 2)) {}
  virtual ~ConnectorTest() { delete connector_; }
  Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, Connector_Lowering_TYPE1>*  connector_;
  Cube<T, TypeParam::LAYOUT>* input_cube;
  Cube<T, Layout_CRDB>* output_cube;
  LoweringConfig * p_config;
};

typedef ::testing::Types<FloatCRDB, FloatBDRC> DataTypes;

TYPED_TEST_CASE(ConnectorTest, DataTypes);

TYPED_TEST(ConnectorTest, TestInitialization) {
  EXPECT_TRUE(this->connector_);
  EXPECT_TRUE(this->input_cube);
  EXPECT_TRUE(this->output_cube);
  EXPECT_EQ(this->p_config->kernel_size, 3);
  EXPECT_EQ(this->input_cube->B, 2);
  EXPECT_EQ(this->input_cube->D, 2);
  EXPECT_EQ(this->input_cube->R, 4);
  EXPECT_EQ(this->input_cube->C, 4);
  EXPECT_EQ(this->input_cube->n_elements, 64);
  EXPECT_EQ(this->output_cube->B, 1);
  EXPECT_EQ(this->output_cube->D, 1);
  EXPECT_EQ(this->output_cube->R, 18);
  EXPECT_EQ(this->output_cube->C, 8);
  EXPECT_EQ(this->output_cube->n_elements, 144);
}

TYPED_TEST(ConnectorTest, TestTransfer){
	for(int i=0; i<this->input_cube->n_elements; i++){
		this->input_cube->p_data[i] = i;
  	}

  	this->connector_->transfer(this->input_cube, this->output_cube);
  	for(int i=0; i<this->output_cube->n_elements; i++){
		EXPECT_EQ(this->output_cube->p_data[i], *this->input_cube->logical_get((i/24)%3+(i/2)%2, (i/8)%3+(i%2), i/72, (i/4)%2));
	}
	
}


TYPED_TEST(ConnectorTest, TestInverseTransfer){
	for(int i=0; i<this->output_cube->n_elements; i++){
		this->output_cube->p_data[i] = i;
  	}


 	typedef typename TypeParam::T T;
  	Cube<T, Layout_CRDB> input_expected(4, 4, 2, 2);

  	this->connector_->inverse_transfer(this->output_cube, this->input_cube);

  	if (TypeParam::LAYOUT == Layout_BDRC){
	  	for(int i=0; i<this->output_cube->n_elements; i++){
			//TODO
		}
	}

	if (TypeParam::LAYOUT == Layout_CRDB){
	  	for(int i=0; i<this->output_cube->n_elements; i++){
	  		*input_expected.logical_get((i/24)%3+(i/2)%2, (i/8)%3+(i%2), i/72, (i/4)%2) += *this->output_cube->logical_get(i/8,i%8,0,0);
		}

		for(int i=0; i<this->input_cube->n_elements; i++){
	  		EXPECT_NEAR(input_expected.p_data[i], this->input_cube->p_data[i],EPS);
		}
	}

	//free(input_expected.p_data);
}

