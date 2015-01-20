#include "../src/Kernel.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include "glog/logging.h"
#include <iostream>
#include <assert.h>
#include <cstring>


typedef ::testing::Types<DataType_SFFloat> DTypes;

template <typename T>
class BlasNNKernelTest : public ::testing::Test {
 public:
  BlasNNKernelTest(){
  	cube1 = new Cube<T, Layout_CRDB>(2, 5, 1, 1);
    cube2 = new Cube<T, Layout_CRDB>(5, 3, 1, 1);
    cube3 = new Cube<T, Layout_CRDB>(2, 3, 1, 1);
  	kernel_ = new Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_NOTRANS>(cube1, cube2, cube3);
  }

  virtual ~BlasNNKernelTest() { delete kernel_; }
  Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_NOTRANS>*  kernel_;
  Cube<T, Layout_CRDB>* cube1;
  Cube<T, Layout_CRDB>* cube2;
  Cube<T, Layout_CRDB>* cube3;
};

TYPED_TEST_CASE(BlasNNKernelTest, DTypes);

TYPED_TEST(BlasNNKernelTest, TestCompute){
	for(int i=0;i<10;i++){
        this->cube1->p_data[i] = 1.0*i;
    }
    for(int i=0;i<15;i++){
        this->cube2->p_data[i] = 3.14/(i+1); 
    }

  this->kernel_->compute(this->cube1, this->cube2, this->cube3);

 	EXPECT_NEAR(this->cube3->p_data[0],3.5903,EPS);
	EXPECT_NEAR(this->cube3->p_data[1],3.16651,EPS);
	EXPECT_NEAR(this->cube3->p_data[2],2.84344,EPS);
	EXPECT_NEAR(this->cube3->p_data[3],28.2358,EPS);   
	EXPECT_NEAR(this->cube3->p_data[4],18.6677,EPS);
	EXPECT_NEAR(this->cube3->p_data[5],14.7929,EPS);
}

template <typename T>
class BlasTNKernelTest : public ::testing::Test {
 protected:;
  BlasTNKernelTest(){
    cube1 = new Cube<T, Layout_CRDB>(5, 2, 1, 1);
    cube2 = new Cube<T, Layout_CRDB>(5, 3, 1, 1);
    cube3 = new Cube<T, Layout_CRDB>(2, 3, 1, 1);
    kernel_ = new Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_TRANS_NOTRANS>(cube1, cube2, cube3);
  }

  virtual ~BlasTNKernelTest() { delete kernel_; }
  Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_TRANS_NOTRANS>*  kernel_;
  Cube<T, Layout_CRDB>* cube1;
  Cube<T, Layout_CRDB>* cube2;
  Cube<T, Layout_CRDB>* cube3;
};

TYPED_TEST_CASE(BlasTNKernelTest, DTypes);

TYPED_TEST(BlasTNKernelTest, TestCompute){
  for(int i=0;i<10;i=i+2){
        this->cube1->p_data[i] = 1.0*i/2;
        this->cube1->p_data[i+1] = 1.0*(i/2+5);
    }
    for(int i=0;i<15;i++){
        this->cube2->p_data[i] = 3.14/(i+1); 
    }

    this->kernel_->compute(this->cube1, this->cube2, this->cube3);

  EXPECT_NEAR(this->cube3->p_data[0],3.5903,EPS);
  EXPECT_NEAR(this->cube3->p_data[1],3.16651,EPS);
  EXPECT_NEAR(this->cube3->p_data[2],2.84344,EPS);
  EXPECT_NEAR(this->cube3->p_data[3],28.2358,EPS);   
  EXPECT_NEAR(this->cube3->p_data[4],18.6677,EPS);
  EXPECT_NEAR(this->cube3->p_data[5],14.7929,EPS);
}

template <typename T>
class BlasNTKernelTest : public ::testing::Test {
 protected:;
  BlasNTKernelTest(){
    cube1 = new Cube<T, Layout_CRDB>(2, 5, 1, 1);
    cube2 = new Cube<T, Layout_CRDB>(3, 5, 1, 1);
    cube3 = new Cube<T, Layout_CRDB>(2, 3, 1, 1);
    kernel_ = new Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_TRANS>(cube1, cube2, cube3);
  }

  virtual ~BlasNTKernelTest() { delete kernel_; }
  Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_TRANS>*  kernel_;
  Cube<T, Layout_CRDB>* cube1;
  Cube<T, Layout_CRDB>* cube2;
  Cube<T, Layout_CRDB>* cube3;
};

TYPED_TEST_CASE(BlasNTKernelTest, DTypes);

TYPED_TEST(BlasNTKernelTest, TestCompute){
  for(int i=0;i<10;i++){
        this->cube1->p_data[i] = 1.0*i;
    }

  for(int i=0;i<15;i++){
      this->cube2->p_data[i] = 3.14/(i+1); 
  }

  this->kernel_->compute(this->cube1, this->cube2, this->cube3);
  EXPECT_NEAR(this->cube3->p_data[0],8.5303,EPS);
  EXPECT_NEAR(this->cube3->p_data[1],3.5362,EPS);
  EXPECT_NEAR(this->cube3->p_data[2],2.2549,EPS);
  EXPECT_NEAR(this->cube3->p_data[3],44.3787,EPS);   
  EXPECT_NEAR(this->cube3->p_data[4],13.6727,EPS);
  EXPECT_NEAR(this->cube3->p_data[5],8.3663,EPS);
}

template <typename T>
class ElemMulKernelTest : public ::testing::Test {
 protected:;
  ElemMulKernelTest(){
    cube1 = new Cube<T, Layout_CRDB>(3, 2, 1, 1);
    cube2 = new Cube<T, Layout_CRDB>(3, 2, 1, 1);
    cube3 = new Cube<T, Layout_CRDB>(3, 2, 1, 1);
    kernel_ = new Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KernelConfig_NONE>(cube1, cube2, cube3);
  }

  virtual ~ElemMulKernelTest() { delete kernel_; }
  Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB,  Kernel_ELEMENTWISEMUL_CPU, KernelConfig_NONE>*  kernel_;
  Cube<T, Layout_CRDB>* cube1;
  Cube<T, Layout_CRDB>* cube2;
  Cube<T, Layout_CRDB>* cube3;
};

TYPED_TEST_CASE(ElemMulKernelTest, DTypes);

TYPED_TEST(ElemMulKernelTest, TestCompute){
  for(int i=0;i<6;i++){
        this->cube1->p_data[i] = i+1;
    }

  for(int i=0;i<6;i++){
      this->cube2->p_data[i] = 3.14/(i+1); 
  }

  this->kernel_->compute(this->cube1, this->cube2, this->cube3);

  for(int i=0;i<6;i++){
    EXPECT_NEAR(this->cube3->p_data[i],3.14,EPS);
  }
}

template <typename T>
class ElemMulTanhKernelTest : public ::testing::Test {
 protected:;
  ElemMulTanhKernelTest(){
    cube1 = new Cube<T, Layout_CRDB>(3, 2, 1, 1);
    cube2 = new Cube<T, Layout_CRDB>(3, 2, 1, 1);
    cube3 = new Cube<T, Layout_CRDB>(3, 2, 1, 1);
    kernel_ = new Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KernelConfig_TANHGRAD_ON_INPUT1>(cube1, cube2, cube3);
  }

  virtual ~ElemMulTanhKernelTest() { delete kernel_; }
  Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB,  Kernel_ELEMENTWISEMUL_CPU, KernelConfig_TANHGRAD_ON_INPUT1>*  kernel_;
  Cube<T, Layout_CRDB>* cube1;
  Cube<T, Layout_CRDB>* cube2;
  Cube<T, Layout_CRDB>* cube3;
};

TYPED_TEST_CASE(ElemMulTanhKernelTest, DTypes);

TYPED_TEST(ElemMulTanhKernelTest, TestCompute){
  for(int i=0;i<6;i++){
        this->cube1->p_data[i] = (i+1)/3.0;
    }

  for(int i=0;i<6;i++){
      this->cube2->p_data[i] = 3.14/(i+1); 
  }

  this->kernel_->compute(this->cube1, this->cube2, this->cube3);

  float actual;
  for(int i=0;i<6;i++){
    actual = 3.14*(1 - pow((i+1),2)/9.0)/(i+1);
    EXPECT_NEAR(this->cube3->p_data[i],actual,EPS);
  }
}