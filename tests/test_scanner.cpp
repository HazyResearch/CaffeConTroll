#include <cstring>
#include <iostream>
#include <assert.h>
#include <cmath>
#include "test_types.h"
#include "gtest/gtest.h"
#include "glog/logging.h"

#include "../src/sched/DeviceDriver_CPU.h"
#include "../src/LogicalCube.h"

using namespace std;

#define LAYOUT TypeParam::LAYOUT
// #define FUNC TypeParam::FUNC

template <typename TypeParam>
class ScannerTANHTest : public ::testing::Test {
  protected:
    typedef typename TypeParam::T T ;

    ScannerTANHTest(){
      cube_ = new LogicalCube<T, LAYOUT>(4, 5, 3, 2);
      scanner_ = new Scanner<T, LAYOUT, FUNC_TANH>(cube_, &pdriver);
    }
    virtual ~ScannerTANHTest() { delete cube_; delete scanner_; }
    Scanner<T, LAYOUT, FUNC_TANH>*  scanner_;
    LogicalCube<T, LAYOUT>* cube_;
    CPUDriver pdriver;
};

typedef ::testing::Types<FloatBDRC, FloatCRDB> DataTypes;

/*
 * Scanner no longer supports TANH; should be its own layer, like ReLU
TYPED_TEST_CASE(ScannerTANHTest, DataTypes);

TYPED_TEST(ScannerTANHTest, TestApply) {

  for(int i=0; i<this->cube_->n_elements; i++){
    this->cube_->get_p_data()[i] = i;
  }
  this->scanner_->apply(this->cube_);

  for(int i=0; i<this->cube_->n_elements; i++){
    EXPECT_NEAR(this->cube_->get_p_data()[i], tanh(i), EPS);
  }
}
*/

template <typename TypeParam>
class ScannerNO_FUNCTest : public ::testing::Test {
  protected:
    typedef typename TypeParam::T T ;

    ScannerNO_FUNCTest(){
      cube_ = new LogicalCube<T, LAYOUT>(4, 5, 3, 2);
      scanner_ = new Scanner<T, LAYOUT, FUNC_NOFUNC>(cube_, &pdriver);
    }
    virtual ~ScannerNO_FUNCTest() { delete cube_; delete scanner_; }
    Scanner<T, LAYOUT, FUNC_NOFUNC>*  scanner_;
    LogicalCube<T, LAYOUT>* cube_;
    CPUDriver pdriver;
};


TYPED_TEST_CASE(ScannerNO_FUNCTest, DataTypes);

TYPED_TEST(ScannerNO_FUNCTest, TestApply) {
  for(int i=0; i<this->cube_->n_elements; i++){
    this->cube_->get_p_data()[i] = i;
  }
  this->scanner_->apply(this->cube_);

  for(int i=0; i<this->cube_->n_elements; i++){
    EXPECT_NEAR(this->cube_->get_p_data()[i], i, EPS);
  }
}
