#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/util.h"
#include "../src/Connector.h"
#include "../src/bridges/ConvolutionBridge.h"
#include "../src/bridges/ParallelizedBridge.h"
#include "../src/sched/DeviceDriver_GPU.h"
#include "test_types.h"
#include "gtest/gtest.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

template <typename TypeParam>
class GPUPerfConvolutionBridgeTest_2 : public ::testing::Test {
  public:
    typedef typename TypeParam::T T;
    GPUPerfConvolutionBridgeTest_2(){
      data1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);
      grad1 = new LogicalCube<T, Layout_CRDB>(iR, iC, iD, mB);

      data2 = new LogicalCube<T, Layout_CRDB>(oR, oC, oD, mB);
      grad2 = new LogicalCube<T, Layout_CRDB> (oR, oC, oD, mB);

      layer1 = new Layer<T, Layout_CRDB>(data1, grad1);
      layer2 = new Layer<T, Layout_CRDB>(data2, grad2);

      cnn::LayerParameter layer_param;
      layer_param.set_gpu_0_batch_proportion(1);
      cnn::ConvolutionParameter * const conv_param = layer_param.mutable_convolution_param();
      conv_param->set_num_output(oD);
      conv_param->set_kernel_size(k);
      conv_param->set_pad(p);
      conv_param->set_stride(s);

      solver_param.set_base_lr(0.01);
      solver_param.set_momentum(0.0);
      solver_param.set_lr_policy("step");
      solver_param.set_stepsize(10000);


      // TODO: set #partition to 8 does not halt
      ParallelizedConvolutionBridge_ = new ParallelizedBridge<DataType_SFFloat,
              ConvolutionBridge>(layer1,
                  layer2, &layer_param, &solver_param, &pdriver, 1, 1);

      ParallelizedConvolutionBridge_->needs_to_calc_backward_grad = true;
    }

    virtual ~GPUPerfConvolutionBridgeTest_2() { delete layer1; delete layer2; delete ParallelizedConvolutionBridge_; }
    ParallelizedBridge<DataType_SFFloat,
              ConvolutionBridge>* ParallelizedConvolutionBridge_;

    LogicalCube<T, Layout_CRDB>* data1;
    LogicalCube<T, Layout_CRDB>* grad1;

    LogicalCube<T, Layout_CRDB>* data2;
    LogicalCube<T, Layout_CRDB>* grad2;

    Layer<T, Layout_CRDB>* layer1;
    Layer<T, Layout_CRDB>* layer2;

    cnn::SolverParameter solver_param;

    CPUDriver pdriver;

    // Lenet L2
    static const int mB = 64;
    static const int iD = 20;
    static const int oD = 50;
    static const int iR = 12;
    static const int iC = 12;
    static const int k = 5;
    static const int s = 1;
    static const int p = 0;

    static const int oR = static_cast<int>((static_cast<float>(iR + 2*p - k) / s)) + 1;
    static const int oC = static_cast<int>((static_cast<float>(iC + 2*p - k) / s)) + 1;
};

typedef ::testing::Types<FloatNOFUNC> DataTypes;

TYPED_TEST_CASE(GPUPerfConvolutionBridgeTest_2, DataTypes);

/*
TYPED_TEST(GPUPerfConvolutionBridgeTest_2, TestForward){

  // Create random data and model parameters
  for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
    this->data1->get_p_data()[i] = float(rand()%100) / 100.0;
    this->grad1->get_p_data()[i] = 0;
  }
  for(int i=0;i<this->k*this->k*this->iD*this->oD;i++){
    this->ParallelizedConvolutionBridge_->p_model_cube->get_p_data()[i] = float(rand()%100) / 100.0;
  }
  for(int i=0;i<this->oD;i++){
    this->ParallelizedConvolutionBridge_->p_bias_cube->get_p_data()[i] = float(rand()%100) / 100.0;
  }

  // Run FW pass many times
  for (int i = 0; i < 100; ++i) {
    this->ParallelizedConvolutionBridge_->forward();
  }
  
  // Print results
  //std::cout<<"\nreport_forward_history\n";
  //this->ParallelizedConvolutionBridge_->_bridges[0]->report_forward_history.print();
  std::cout<<"\nreport_forward_lowering\n";
  this->ParallelizedConvolutionBridge_->_bridges[0]->report_forward_lowering.print();
  //std::cout<<"\nreport_forward_kernel\n";
  //this->ParallelizedConvolutionBridge_->_bridges[0]->report_forward_kernel.print();
}
*/


TYPED_TEST(GPUPerfConvolutionBridgeTest_2, TestForwardBackward){

  // Create random data and model parameters
  for(int i=0;i<this->iR*this->iC*this->iD*this->mB;i++){
    this->data1->get_p_data()[i] = float(rand()%100) / 100.0;
    this->grad1->get_p_data()[i] = 0;
  }
  for(int i=0;i<this->k*this->k*this->iD*this->oD;i++){
    this->ParallelizedConvolutionBridge_->p_model_cube->get_p_data()[i] = float(rand()%100) / 100.0;
  }
  for(int i=0;i<this->oD;i++){
    this->ParallelizedConvolutionBridge_->p_bias_cube->get_p_data()[i] = float(rand()%100) / 100.0;
  }
  for (int i=0;i<this->oR*this->oC*this->oD*this->mB;i++) {
    this->data2->get_p_data()[i] = 0;
    this->grad2->get_p_data()[i] = i*0.1;
  }

  this->ParallelizedConvolutionBridge_->force_host_to_device_model_copy();
  this->ParallelizedConvolutionBridge_->force_host_to_device_bias_copy();

  // Run FW and BW pass many times
  for (int i = 0; i < 100; ++i) {
    this->ParallelizedConvolutionBridge_->forward();
    //this->ParallelizedConvolutionBridge_->report_forward();
    this->ParallelizedConvolutionBridge_->backward();
    //this->ParallelizedConvolutionBridge_->report_backward();
  }
  
  this->ParallelizedConvolutionBridge_->force_device_to_host_model_copy();
  this->ParallelizedConvolutionBridge_->force_device_to_host_bias_copy();
  
  // Print results
  std::cout<<"\n\nreport_pbridge_fw\n";
  this->ParallelizedConvolutionBridge_->report_forward_history.print();
  std::cout<<"\nreport_pbridge_bw\n";
  this->ParallelizedConvolutionBridge_->report_backward_updateweight_history.print();
  std::cout << "\nFor bridge 0 of total " << this->ParallelizedConvolutionBridge_->_gpu_bridges.size() << " bridges:\n";
  std::cout<<"\nreport_forward_history\n";
  this->ParallelizedConvolutionBridge_->_gpu_bridges[0]->report_forward_history.print();
  std::cout<<"\nreport_backward_history\n";
  this->ParallelizedConvolutionBridge_->_gpu_bridges[0]->report_backward_updateweight_history.print();
}
