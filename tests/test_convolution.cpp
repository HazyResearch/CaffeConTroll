
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>
#include "test_types.h"
#include "gtest/gtest.h"

#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/ConvolutionBridge.h"

#ifdef _GPU_TARGET
#include "../src/sched/DeviceDriver_GPU.h"
#endif

#include "../src/sched/DeviceDriver_CPU.h"

#include "../src/kernels/include.h"


template <typename T>
void simple_conv(LogicalCube<T, Layout_CRDB>* in, LogicalCube<T, Layout_CRDB>* kernel, LogicalCube<T, Layout_CRDB>* out){
  int ofm = out->D;
  int ifm = in->D;
  for (int n = 0; n < out->B; n++) {
    for (int o = 0; o < ofm; o++) {
      for (int k = 0; k < ifm; k++) {
        for (int y = 0; y < out->R; y++) {
          for (int x = 0; x < out->C; x++) {
            for (int p = 0; p < kernel->R; p++) {
              for (int q = 0; q < kernel->C; q++) {
                int in_y = y + p;
                int in_x = x + q;
                *out->logical_get(y, x, o, n) +=
                  *in->logical_get(in_y, in_x, k, n)*
                  *kernel->logical_get(p, q, k, o);
              }
            }
          }
        }
      }
    }
  }
}

template <typename TypeParam>
class ParallelizedConvolutionBridgeTest : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    ParallelizedConvolutionBridgeTest(){
      data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2 = new LogicalCube<T, Layout_CRDB>(oR, oC, oD, mB);
      grad2 = new LogicalCube<T, Layout_CRDB> (oR, oC, oD, mB);

      layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
      layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

      cnn::LayerParameter layer_param;
      cnn::ConvolutionParameter * const conv_param = layer_param.mutable_convolution_param();
      conv_param->set_num_output(oD);
      conv_param->set_kernel_size(k);
      conv_param->set_pad(p);
      conv_param->set_stride(s);

      solver_param.set_base_lr(0.01);
      solver_param.set_momentum(0.0);
      solver_param.set_lr_policy("step");
      solver_param.set_stepsize(10000);

      ConvolutionBridge_ = new ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, 
              FUNC_NOFUNC, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>
              (layer1, layer2, &layer_param, &solver_param, &pdriver);

      ConvolutionBridge_->run_with_n_threads = 1;

      ConvolutionBridge_->needs_to_calc_backward_grad = true;
    }

    virtual ~ParallelizedConvolutionBridgeTest() { delete layer1; delete layer2; }
    
    ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, DataType_SFFloat,
      Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver> * ConvolutionBridge_;

    LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;

    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    cnn::SolverParameter solver_param;

    CPUDriver pdriver;

    /*
    static const int mB = 4;
    static const int iD = 3;
    static const int oD = 10;
    static const int iR = 20;
    static const int iC = 20;
    static const int k = 5;
    static const int s = 2;
    static const int p = 2;
    */

    static const int mB = 256;
    static const int iD = 48;
    static const int oD = 128;
    static const int iR = 27;
    static const int iC = 27;
    static const int k = 5;
    static const int s = 1;
    static const int p = 2;

    static const int oR = static_cast<int>((static_cast<float>(iR + 2*p - k) / s)) + 1;
    static const int oC = static_cast<int>((static_cast<float>(iC + 2*p - k) / s)) + 1;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(ParallelizedConvolutionBridgeTest, DataTypes);

TYPED_TEST(ParallelizedConvolutionBridgeTest, TestInitialization){
  EXPECT_TRUE(this->ConvolutionBridge_);
  EXPECT_TRUE(this->layer1);
  EXPECT_TRUE(this->layer2);
}

TYPED_TEST(ParallelizedConvolutionBridgeTest, TestForward){

  this->ConvolutionBridge_->forward();

  this->ConvolutionBridge_->report_forward_last_transfer.print();
  this->ConvolutionBridge_->report_forward_kernel.print();
  this->ConvolutionBridge_->report_forward_lowering.print();

}
