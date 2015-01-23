///// NOT REQUIRED for now--- UPDATE after rest of the tests are done

#include <cstring>
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "../src/LogicalCube.h"
#include <iostream>
#include <assert.h>
#include "test_types.h"

using namespace std;

template <typename TypeParam>
class LogicalCubeTest : public ::testing::Test {
 protected:
 	typedef typename TypeParam::T T ;
  LogicalCubeTest()
      : cube_(new LogicalCube<T, TypeParam::LAYOUT>(5, 4, 3, 2)) {}
  virtual ~LogicalCubeTest() { delete cube_; }
  LogicalCube<T, TypeParam::LAYOUT>*  cube_;
};

typedef ::testing::Types<FloatCRDB, FloatBDRC> DataTypes;

TYPED_TEST_CASE(LogicalCubeTest, DataTypes);

TYPED_TEST(LogicalCubeTest, TestInitialization) {	
  EXPECT_TRUE(this->cube_);
  EXPECT_EQ(this->cube_->B, 2);
  EXPECT_EQ(this->cube_->D, 3);
  EXPECT_EQ(this->cube_->R, 5);
  EXPECT_EQ(this->cube_->C, 4);
  EXPECT_EQ(this->cube_->n_elements, 120);
}

TYPED_TEST(LogicalCubeTest, TestLogicalFetcher) {

  for(int i=0; i<this->cube_->n_elements; i++){
		this->cube_->p_data[i] = i;
  }
  typedef typename TypeParam::T T;
  T * dataptr;
  if (TypeParam::LAYOUT == Layout_BDRC){
	  for(int r=0;r<this->cube_->R;r++){
			for(int c=0;c<this->cube_->C;c++){
				for(int d=0;d<this->cube_->D;d++){
					for(int b=0;b<this->cube_->B;b++){
						dataptr = this->cube_->logical_get(r,c,d,b);	
						int actual = this->cube_->p_data[b + d*this->cube_->B + r*this->cube_->B*this->cube_->D + c*this->cube_->B*this->cube_->R*this->cube_->D];
						EXPECT_EQ(*dataptr, actual);
					}	
				}		
			}
		}
   }

   if (TypeParam::LAYOUT == Layout_CRDB){
	  for(int r=0;r<this->cube_->R;r++){
			for(int c=0;c<this->cube_->C;c++){
				for(int d=0;d<this->cube_->D;d++){
					for(int b=0;b<this->cube_->B;b++){
						dataptr = this->cube_->logical_get(r,c,d,b);	
						int actual = this->cube_->p_data[c + r*this->cube_->C + d*this->cube_->R*this->cube_->C + b*this->cube_->R*this->cube_->C*this->cube_->D];
						EXPECT_EQ(*dataptr, actual);
					}		
				}		
			}
		}
   }	  
}

TYPED_TEST(LogicalCubeTest, TestLogicalMatrix) {
  for(int i=0; i<this->cube_->n_elements; i++){
		this->cube_->p_data[i] = i;
  }
  typedef typename TypeParam::T T;

  const int k = 3;
  const int s = 1;
  for(int b=0; b<this->cube_->B; b++){
  	for(int d=0; d<this->cube_->D; d++){
  		const LogicalMatrix<T> m = this->cube_->get_logical_matrix(d, b);
  		this->cube_->template lower_logical_matrix<LOWERING_TYPE1>(&m, b, d, k, s);		
  	}
  }
}

// Testing RCD Slice -- implemented only for CRDB Layout
class LogicalCubeTest_CRDB : public ::testing::Test {
 protected:
  LogicalCubeTest_CRDB()
      : cube_(new LogicalCube<DataType_SFFloat, Layout_CRDB>(4, 5, 3, 2)) {}
  virtual ~LogicalCubeTest_CRDB() { delete cube_; }
  LogicalCube<DataType_SFFloat, Layout_CRDB>*  cube_;
};

TEST_F(LogicalCubeTest_CRDB, TestRCDSlice){
	for(int i=0; i<this->cube_->n_elements; i++){
		this->cube_->p_data[i] = i;
  	}
  	
  	DataType_SFFloat * dataptr;

  	for(int b=0;b<this->cube_->B;b++){
		dataptr = this->cube_->physical_get_RCDslice(b);	
		assert(*dataptr == this->cube_->p_data[b*this->cube_->R*this->cube_->C*this->cube_->D]);		
	}	
}

