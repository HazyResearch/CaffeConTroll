#include <cstring>
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "../src/Cube.h"
#include <iostream>
#include <assert.h>
#include "test_types.h"

using namespace std;

template <typename TypeParam>
class CubeTest : public ::testing::Test {
 protected:
 	typedef typename TypeParam::T T ;
  CubeTest()
      : cube_(new Cube<T, TypeParam::LAYOUT>(4, 5, 3, 2)) {}
  virtual ~CubeTest() { delete cube_; }
  Cube<T, TypeParam::LAYOUT>*  cube_;
};

typedef ::testing::Types<FloatCRDB, FloatBDRC> DataTypes;

TYPED_TEST_CASE(CubeTest, DataTypes);

TYPED_TEST(CubeTest, TestInitialization) {

  EXPECT_TRUE(this->cube_);
  EXPECT_EQ(this->cube_->B, 2);
  EXPECT_EQ(this->cube_->D, 3);
  EXPECT_EQ(this->cube_->R, 4);
  EXPECT_EQ(this->cube_->C, 5);
  EXPECT_EQ(this->cube_->n_elements, 120);
}

TYPED_TEST(CubeTest, TestLogicalFetcher) {

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

class CubeTest_CRDB : public ::testing::Test {
 protected:
  CubeTest_CRDB()
      : cube_(new Cube<DataType_SFFloat, Layout_CRDB>(4, 5, 3, 2)) {}
  virtual ~CubeTest_CRDB() { delete cube_; }
  Cube<DataType_SFFloat, Layout_CRDB>*  cube_;
};

TEST_F(CubeTest_CRDB, TestRCSlice){
	for(int i=0; i<this->cube_->n_elements; i++){
		this->cube_->p_data[i] = i;
  	}
  	
  	DataType_SFFloat * dataptr;

  	for(int b=0;b<this->cube_->B;b++){
		for(int d=0;d<this->cube_->D;d++){
			dataptr = this->cube_->physical_get_RCslice(d,b);	
			assert(*dataptr == this->cube_->p_data[d*this->cube_->R*this->cube_->C + b*this->cube_->R*this->cube_->C*this->cube_->D]);
		}	
	}	
}

