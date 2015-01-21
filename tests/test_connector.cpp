#include "../src/Connector.h"
#include "../src/LogicalCube.h"
#include "test_types.h"
#include <cstring>
#include "gtest/gtest.h"
#include "glog/logging.h"
#include <iostream>
#include <assert.h>
using namespace std;

template<typename T, LayoutType LAYOUT>
void simple_lowering(LogicalCube<T, LAYOUT>* in, LogicalCube<T, Layout_CRDB>* out, int k, int s){
  int outc,outr=0;
  for(size_t kd=0;kd<in->D;kd++) {
    for(size_t kr=0;kr<k;kr++) {
      for(size_t kc=0;kc<k;kc++) {
        outc = 0;
        for(size_t ib=0;ib<in->B;ib++) {
          for(size_t cr=0;cr<(in->R-k)/s+1;cr++) {
            for(size_t cc=0;cc<(in->C-k)/s+1;cc++) {
                *out->logical_get(outr, outc, 0, 0) = 
              *in->logical_get(cr*s+kr, cc*s+kc, kd, ib);
              outc ++;
            }
          }
        }
        outr ++;
      }
    }
  }
}

template <typename TypeParam>
class ConnectorTest : public ::testing::Test {
 protected:
 	typedef typename TypeParam::T T;
  ConnectorTest(){
  	input_cube = new LogicalCube<T, TypeParam::LAYOUT>(R,C,D,B);
 	p_config = new LoweringConfig();
	p_config->kernel_size = k;
  p_config->stride = s;

	output_cube = new LogicalCube<T, Layout_CRDB>(k*k*D,((R-k)/s+1)*((C-k)/s+1)*B,1,1);
  	connector_ = new Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, LOWERING_TYPE1>(input_cube, output_cube, 
		p_config);	
  }
      //: connector_(new Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, LOWERING_TYPE1>(4, 5, 3, 2)) {}
  virtual ~ConnectorTest() { delete connector_; }
  Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, LOWERING_TYPE1>*  connector_;
  LogicalCube<T, TypeParam::LAYOUT>* input_cube;
  LogicalCube<T, Layout_CRDB>* output_cube;
  LoweringConfig * p_config;
  const int R = 4;
  const int C = 4;
  const int D = 2;
  const int B = 2;
  const int k = 3;
  const int s = 1;
};


//TODO -- Check For Layout Type BDRC-- not working properly
typedef ::testing::Types<FloatCRDB> DataTypes;

TYPED_TEST_CASE(ConnectorTest, DataTypes);

TYPED_TEST(ConnectorTest, TestInitialization) {
  EXPECT_TRUE(this->connector_);
}

TYPED_TEST(ConnectorTest, TestLowering){
 typedef typename TypeParam::T T;
 int R = this->R;
 int C = this->C;
 int D = this->D;
 int B = this->B;
 int k = this->k;
 int s = this->s;
 int oR = (R-k)/s+1;
 int oC = (C-k)/s+1;

	for(int r=0;r<this->input_cube->R;r++){
      for(int c=0;c<this->input_cube->C;c++){
        for(int d=0;d<this->input_cube->D;d++){
          for(int b=0;b<this->input_cube->B;b++){
            *this->input_cube->logical_get(r,c,d,b) = rand()%10;
          } 
        }   
      }
    }

    LogicalCube<T, Layout_CRDB>* expected_output = new LogicalCube<T, Layout_CRDB>(k*k*D,oR*oC*B,1,1);
    simple_lowering<T,TypeParam::LAYOUT>(this->input_cube,expected_output,k,s);

    int n = this->output_cube->n_elements; 
  	this->connector_->lower_cube(this->input_cube, this->output_cube);

  	for(int i=0; i<n; i++){
      EXPECT_NEAR(this->output_cube->p_data[i],  expected_output->p_data[i],EPS);
	}
}

//TODO -- Check for inverse lowering
/*
TYPED_TEST(ConnectorTest, TestInverseLowering){
	for(int i=0; i<this->output_cube->n_elements; i++){
		this->output_cube->p_data[i] = i;
  	}


 	typedef typename TypeParam::T T;
  	LogicalCube<T, Layout_CRDB> input_expected(4, 4, 2, 2);

  	this->connector_->inverse_lower_cube(this->output_cube, this->input_cube);

	if (TypeParam::LAYOUT == Layout_CRDB){
	  	for(int i=0; i<this->output_cube->n_elements; i++){
	  		*input_expected.logical_get((i/24)%3+(i/2)%2, (i/8)%3+(i%2), i/72, (i/4)%2) += *this->output_cube->logical_get(i/8,i%8,0,0);
		}

		for(int i=0; i<this->input_cube->n_elements; i++){
	  		//EXPECT_NEAR(input_expected.p_data[i], this->input_cube->p_data[i],EPS);
		}
	}
}
*/

