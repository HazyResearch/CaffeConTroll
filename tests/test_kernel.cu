#include "../src/Kernel.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include "glog/logging.h"
#include <iostream>
#include <assert.h>
#include <cstring>
#include "../src/sched/DeviceDriver_GPU.h"

template <typename T>
void simple_mult(LogicalCube<T, Layout_CRDB>* in1, LogicalCube<T, Layout_CRDB>* in2, LogicalCube<T, Layout_CRDB>* out){
  for(int i=0;i<in1->R;i++){
    for(int j=0;j<in2->C;j++)
    {
        *out->logical_get(i, j, 0, 0)=0;
        for(int k=0;k<in1->C;k++)
        {
            *out->logical_get(i, j, 0, 0) += *in1->logical_get(i, k, 0, 0) * (*in2->logical_get(k, j, 0, 0));
        }
    }
  }
}

template <typename T>
void simple_mult2(LogicalCube<T, Layout_CRDB>* in1, LogicalCube<T, Layout_CRDB>* in2, LogicalCube<T, Layout_CRDB>* out){
  for(int i=0;i<in1->C;i++){
    for(int j=0;j<in2->C;j++)
    {
        *out->logical_get(i, j, 0, 0)=0;
        for(int k=0;k<in1->R;k++)
        {
            *out->logical_get(i, j, 0, 0) += *in1->logical_get(k, i, 0, 0) * (*in2->logical_get(k, j, 0, 0));
        }
    }
  }
}

template <typename T>
void simple_mult3(LogicalCube<T, Layout_CRDB>* in1, LogicalCube<T, Layout_CRDB>* in2, LogicalCube<T, Layout_CRDB>* out){
  for(int i=0;i<in1->R;i++){
    for(int j=0;j<in2->R;j++)
    {
        *out->logical_get(i, j, 0, 0)=0;
        for(int k=0;k<in1->C;k++)
        {
            *out->logical_get(i, j, 0, 0) += *in1->logical_get(i, k, 0, 0) * (*in2->logical_get(j, k, 0, 0));
        }
    }
  }
}

typedef ::testing::Types<FloatCRDB> DTypes;

template <typename TypeParam>
class BlasNNKernelTest : public ::testing::Test {
 protected:
  typedef typename TypeParam::T T;

  GPUDriver driver;

  BlasNNKernelTest(){
  	cube1 = new LogicalCube<T, Layout_CRDB>(i1R, i1C, 1, 1, &driver);
    cube2 = new LogicalCube<T, Layout_CRDB>(i1C, i2C, 1, 1, &driver);
    cube3 = new LogicalCube<T, Layout_CRDB>(i1R, i2C, 1, 1, &driver);

  	kernel_ = new Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, 
      KernelConfig_GEMM_NOTRANS_NOTRANS>(cube1, cube2, cube3, &driver);
  }

  virtual ~BlasNNKernelTest() { delete kernel_; }
  Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_NOTRANS>*  kernel_;
  LogicalCube<T, Layout_CRDB>* cube1;
  LogicalCube<T, Layout_CRDB>* cube2;
  LogicalCube<T, Layout_CRDB>* cube3;
  const int i1R = 10;
  const int i1C = 8;
  const int i2C = 15;
};

TYPED_TEST_CASE(BlasNNKernelTest, DTypes);

TYPED_TEST(BlasNNKernelTest, TestCompute){
  typedef typename TypeParam::T T;

  float * a = new float [this->i1R*this->i1C];
  float * b = new float [this->i1C*this->i2C];
  float * c = new float [this->i1R*this->i2C];

	for(int i=0;i<this->i1R*this->i1C;i++){
        a[i] = (rand() % 100)/10.0;
    }
  for(int i=0;i<this->i1C*this->i2C;i++){
        b[i] = (rand() % 100)/10.0;
    }

  int ct = 0;
  for(int r=0;r<this->i1R;r++){
    for(int cc=0;cc<this->i1C;cc++){
      std::cout << a[ct++] << "   ";
    }
    std::cout << std::endl;
  }
  std::cout << "\n\n\n" << std::endl;

  ct = 0;
  for(int r=0;r<this->i1C;r++){
    for(int cc=0;cc<this->i2C;cc++){
      std::cout << b[ct++] << "   ";
    }
    std::cout << std::endl;
  }
  std::cout << "\n\n\n" << std::endl;

  cudaMemcpy(this->cube1->get_p_data(), a, this->i1R*this->i1C*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(this->cube2->get_p_data(), b, this->i1C*this->i2C*sizeof(float), cudaMemcpyHostToDevice);

  LogicalCube<T, Layout_CRDB>* out_expected = new LogicalCube<T, Layout_CRDB>(this->i1R, this->i2C, 1, 1);

  this->kernel_->compute(this->cube1, this->cube2, this->cube3);

  cudaMemcpy(c, this->cube3->get_p_data(), this->i1R*this->i2C*sizeof(float), cudaMemcpyDeviceToHost);

  ct = 0;
  for(int r=0;r<this->i1R;r++){
    for(int cc=0;cc<this->i2C;cc++){
      std::cout << c[ct++] << "   ";
    }
    std::cout << std::endl;
  }

  for(int i=0; i<6; i++){
    EXPECT_NEAR(c[i], out_expected->get_p_data()[i], EPS);
  }
}


/*
template <typename TypeParam>
class ElemMulKernelTest : public ::testing::Test {
 protected:
  typedef typename TypeParam::T T;
  ElemMulKernelTest(){
    cube1 = new LogicalCube<T, Layout_CRDB>(i1R, i1C, 1, 1);
    cube2 = new LogicalCube<T, Layout_CRDB>(i1R, i1C, 1, 1);
    cube3 = new LogicalCube<T, Layout_CRDB>(i1R, i1C, 1, 1);
    kernel_ = new Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KernelConfig_NONE>(cube1, cube2, cube3);
  }

  virtual ~ElemMulKernelTest() { delete kernel_; }
  Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB,  Kernel_ELEMENTWISEMUL_CPU, KernelConfig_NONE>*  kernel_;
  LogicalCube<T, Layout_CRDB>* cube1;
  LogicalCube<T, Layout_CRDB>* cube2;
  LogicalCube<T, Layout_CRDB>* cube3;
  const int i1R = 10;
  const int i1C = 8;
};

TYPED_TEST_CASE(ElemMulKernelTest, DTypes);

TYPED_TEST(ElemMulKernelTest, TestCompute){
  float expected;
  for(int i=0;i<this->i1R*this->i1C;i++){
        this->cube1->get_p_data()[i] = (rand() % 100)/10.0;
    }

  for(int i=0;i<this->i1R*this->i1C;i++){
      this->cube2->get_p_data()[i] = (rand() % 100)/10.0;
  }

  this->kernel_->compute(this->cube1, this->cube2, this->cube3);

  for(int i=0;i<this->i1R*this->i1C;i++){
    expected = this->cube1->get_p_data()[i] * this->cube2->get_p_data()[i];
    EXPECT_NEAR(this->cube3->get_p_data()[i],expected,EPS);
  }
}

template <typename TypeParam>
class ElemMulTanhKernelTest : public ::testing::Test {
 typedef typename TypeParam::T T;
 protected:;
  ElemMulTanhKernelTest(){
    cube1 = new LogicalCube<T, Layout_CRDB>(i1R, i1C, 1, 1);
    cube2 = new LogicalCube<T, Layout_CRDB>(i1R, i1C, 1, 1);
    cube3 = new LogicalCube<T, Layout_CRDB>(i1R, i1C, 1, 1);
    kernel_ = new Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KernelConfig_TANHGRAD_ON_INPUT1>(cube1, cube2, cube3);
  }

  virtual ~ElemMulTanhKernelTest() { delete kernel_; }
  Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB,  Kernel_ELEMENTWISEMUL_CPU, KernelConfig_TANHGRAD_ON_INPUT1>*  kernel_;
  LogicalCube<T, Layout_CRDB>* cube1;
  LogicalCube<T, Layout_CRDB>* cube2;
  LogicalCube<T, Layout_CRDB>* cube3;
  const int i1R = 10;
  const int i1C = 8;
};

TYPED_TEST_CASE(ElemMulTanhKernelTest, DTypes);

TYPED_TEST(ElemMulTanhKernelTest, TestCompute){
  float expected;
  for(int i=0;i<this->i1R*this->i1C;i++){
        this->cube1->get_p_data()[i] = (rand() % 100)/10.0;
    }

  for(int i=0;i<this->i1R*this->i1C;i++){
      this->cube2->get_p_data()[i] = (rand() % 100)/10.0;
  }

  this->kernel_->compute(this->cube1, this->cube2, this->cube3);

  for(int i=0;i<this->i1R*this->i1C;i++){
    expected = this->cube2->get_p_data()[i]*(1 - pow(this->cube1->get_p_data()[i],2));
    EXPECT_NEAR(this->cube3->get_p_data()[i],expected,EPS);
  }
}

template <typename TypeParam>
class BlasTNKernelTest : public ::testing::Test {
 protected:
  typedef typename TypeParam::T T;
  BlasTNKernelTest(){
    cube1 = new LogicalCube<T, Layout_CRDB>(5, 2, 1, 1);
    cube2 = new LogicalCube<T, Layout_CRDB>(5, 3, 1, 1);
    cube3 = new LogicalCube<T, Layout_CRDB>(2, 3, 1, 1);
    kernel_ = new Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_TRANS_NOTRANS>(cube1, cube2, cube3);
  }

  virtual ~BlasTNKernelTest() { delete kernel_; }
  Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_TRANS_NOTRANS>*  kernel_;
  LogicalCube<T, Layout_CRDB>* cube1;
  LogicalCube<T, Layout_CRDB>* cube2;
  LogicalCube<T, Layout_CRDB>* cube3;
};

TYPED_TEST_CASE(BlasTNKernelTest, DTypes);

TYPED_TEST(BlasTNKernelTest, TestCompute){
  typedef typename TypeParam::T T;
  for(int i=0;i<10;i++){
        this->cube1->get_p_data()[i] = 1.0*i;
    }
    for(int i=0;i<15;i++){
        this->cube2->get_p_data()[i] = 3.14/(i+1);
    }

  LogicalCube<T, Layout_CRDB>* out_expected = new LogicalCube<T, Layout_CRDB>(2, 3, 1, 1);

  this->kernel_->compute(this->cube1, this->cube2, this->cube3);
  simple_mult2<T>(this->cube1,this->cube2,out_expected);

  for(int i=0; i<6; i++){
    EXPECT_NEAR(this->cube3->get_p_data()[i], out_expected->get_p_data()[i], EPS);
  }
}


template <typename TypeParam>
class BlasNTKernelTest : public ::testing::Test {
 protected:
  typedef typename TypeParam::T T;
  BlasNTKernelTest(){
    cube1 = new LogicalCube<T, Layout_CRDB>(2, 5, 1, 1);
    cube2 = new LogicalCube<T, Layout_CRDB>(3, 5, 1, 1);
    cube3 = new LogicalCube<T, Layout_CRDB>(2, 3, 1, 1);
    kernel_ = new Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_TRANS>(cube1, cube2, cube3);
  }

  virtual ~BlasNTKernelTest() { delete kernel_; }
  Kernel<T, Layout_CRDB, T, Layout_CRDB, T, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_TRANS>*  kernel_;
  LogicalCube<T, Layout_CRDB>* cube1;
  LogicalCube<T, Layout_CRDB>* cube2;
  LogicalCube<T, Layout_CRDB>* cube3;
};

TYPED_TEST_CASE(BlasNTKernelTest, DTypes);

TYPED_TEST(BlasNTKernelTest, TestCompute){
  typedef typename TypeParam::T T;
  for(int i=0;i<10;i++){
        this->cube1->get_p_data()[i] = 1.0*i;
    }
    for(int i=0;i<15;i++){
        this->cube2->get_p_data()[i] = 3.14/(i+1);
    }

  LogicalCube<T, Layout_CRDB>* out_expected = new LogicalCube<T, Layout_CRDB>(2, 3, 1, 1);

  this->kernel_->compute(this->cube1, this->cube2, this->cube3);
  simple_mult3<T>(this->cube1,this->cube2,out_expected);

  for(int i=0; i<6; i++){
    EXPECT_NEAR(this->cube3->get_p_data()[i], out_expected->get_p_data()[i], EPS);
  }
}
*/
