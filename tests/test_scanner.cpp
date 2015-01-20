#include <cstring>
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "../src/Cube.h"
#include <iostream>
#include <assert.h>
#include <cmath>
#include "test_types.h"

using namespace std;

#define LAYOUT TypeParam::LAYOUT
// #define FUNC TypeParam::FUNC

template <typename TypeParam>
class ScannerTANHTest : public ::testing::Test {
 protected:
	typedef typename TypeParam::T T ;

	ScannerTANHTest()
			: cube_(new Cube<T, LAYOUT>(4, 5, 3, 2)),
				scanner_(new Scanner<T, LAYOUT, FUNC_TANH>(cube_)) {}
	virtual ~ScannerTANHTest() { delete cube_; delete scanner_; }
	Scanner<T, LAYOUT, FUNC_TANH>*  scanner_;
	Cube<T, LAYOUT>* cube_;
};

typedef ::testing::Types<FloatBDRC, FloatCRDB> DataTypes;

TYPED_TEST_CASE(ScannerTANHTest, DataTypes);

TYPED_TEST(ScannerTANHTest, TestApply) {

	for(int i=0; i<this->cube_->n_elements; i++){
		this->cube_->p_data[i] = i;
	}
	this->scanner_->apply(this->cube_);
	typedef typename TypeParam::T T;
	float ind=0;
	for(int i=0; i<this->cube_->n_elements; i++){
		EXPECT_NEAR(this->cube_->p_data[i], tanh(i), EPS);
	}   
}

template <typename TypeParam>
class ScannerNO_FUNCTest : public ::testing::Test {
 protected:
	typedef typename TypeParam::T T ;

	ScannerNO_FUNCTest()
			: cube_(new Cube<T, LAYOUT>(4, 5, 3, 2)),
				scanner_(new Scanner<T, LAYOUT, FUNC_NOFUNC>(cube_)) {}
	virtual ~ScannerNO_FUNCTest() { delete cube_; delete scanner_; }
	Scanner<T, LAYOUT, FUNC_NOFUNC>*  scanner_;
	Cube<T, LAYOUT>* cube_;
};


TYPED_TEST_CASE(ScannerNO_FUNCTest, DataTypes);

TYPED_TEST(ScannerNO_FUNCTest, TestApply) {
	for(int i=0; i<this->cube_->n_elements; i++){
		this->cube_->p_data[i] = i;
	}
	this->scanner_->apply(this->cube_);
	typedef typename TypeParam::T T;
	float ind=0;
	for(int i=0; i<this->cube_->n_elements; i++){
		EXPECT_NEAR(this->cube_->p_data[i], i, EPS);
	}   
}