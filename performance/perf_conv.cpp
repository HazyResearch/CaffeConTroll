#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/ConvolutionBridge.h"
#include "../src/Report.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

void PRINT_CONFIG(int iR, int iC, int iD, int oD, int k, int mB){
    std::cout << "Input = " << iR << " x " << iC <<" ";
    std::cout << " Kernel = " << k << " x " << k << " ";
    std::cout << " Channels: " << "Input: " << iD << " Output: " << oD << " ";
    std::cout << " BatchSize = " << mB << std::endl;
}

void PERF_CONVOLUTION_BRIDGE(int iR, int iC, int iD, int oD, int k, int mB) {
    // const int mB = 10;
    // const int iD = 5;
    // const int oD = 5;
    // const int iR = 22;
    // const int iC = 22;
    // const int k = 2;
    const int s = 2;
    const int p = 0;
    const int oR = static_cast<int>(floor(static_cast<float>(iR + 2*p - k) / s)) + 1;
    const int oC = static_cast<int>(floor(static_cast<float>(iC + 2*p - k) / s)) + 1;
    const InitializerType weight_initializer = XAVIER;
    const InitializerType bias_initializer = CONSTANT;

    LogicalCube<float, Layout_CRDB>* data1 = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
    LogicalCube<float, Layout_CRDB>* grad1 = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
    
    LogicalCube<float, Layout_CRDB>* data2 = new LogicalCube<float, Layout_CRDB>(oR, oC, oD, mB);
    LogicalCube<float, Layout_CRDB>* grad2 = new LogicalCube<float, Layout_CRDB> (oR, oC, oD, mB);

    Layer<float, Layout_CRDB>* layer1 = new Layer<float, Layout_CRDB>(data1, grad1);
    Layer<float, Layout_CRDB>* layer2 = new Layer<float, Layout_CRDB>(data2, grad2);

    BridgeConfig * bconfig = new BridgeConfig(k, oD, p, s, true, weight_initializer, bias_initializer);

    ConvolutionBridge< CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, float, Layout_CRDB, float, Layout_CRDB>* ConvolutionBridge_;
    ConvolutionBridge_ = new ConvolutionBridge< CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, float, Layout_CRDB, float, Layout_CRDB>(layer1, layer2, bconfig); 

    for(int i=0;i<iR*iC*iD*mB;i++){
        data1->p_data[i] = 0.1*(rand()%10);
    }

    for(int i=0;i<k*k*iD*oD;i++){
        ConvolutionBridge_->model_cube()->p_data[i] = 0.01*(rand()%10);
    }
    
    for(int i=0;i<oD;i++){
        ConvolutionBridge_->bias_cube()->p_data[i] = 0.1*(rand()%10);
    }

    ConvolutionBridge_->forward();
    cout << "Forward Report - " << endl;   

    cout << "Lowering Report - " << endl;
    ConvolutionBridge_->report_forward_lowering.print();
    cout << "GEMM Kernel Report - " << endl;
    ConvolutionBridge_->report_forward_kernel.print();
    cout << "Total Report - " << endl;
    ConvolutionBridge_->report_forward_history.print();

    for(int i=0;i<oR*oC*oD*mB;i++){
        grad2->p_data[i] = i*0.1;
    }
    
    for(int i=0;i<iR*iC*iD*mB;i++){
        grad1->p_data[i] = 0;
    }
    
    ConvolutionBridge_->backward();

    cout << "Backward Report - " << endl;

    cout << "Inverse Lowering Report - " << endl;
    ConvolutionBridge_->report_backward_inverse_lowering.print();
    cout << "Weight Update Report - " << endl;
    ConvolutionBridge_->report_backward_weight_kernel.print();
    cout << "Gradient Update Report - " << endl;
    ConvolutionBridge_->report_backward_grad_kernel.print();
    cout << "Total Report - " << endl;
    ConvolutionBridge_->report_backward_updateweight_history.print();
    cout << endl;
    
    delete layer1; delete layer2; delete bconfig;
    delete data1; delete data2; delete grad1; delete grad2;
}

int main(int argc, const char * argv[]) {
  
  PRINT_CONFIG(16,16,32,96,5,64);
  PERF_CONVOLUTION_BRIDGE(16,16,32,96,5,64);

  // PRINT_CONFIG(32,32,32,96,5,64);
  // PERF_CONVOLUTION_BRIDGE(32,32,32,96,5,64);

  // PRINT_CONFIG(64,64,32,96,5,64);
  // PERF_CONVOLUTION_BRIDGE(64,64,32,96,5,64);

  // PRINT_CONFIG(128,128,32,96,5,64);
  // PERF_CONVOLUTION_BRIDGE(128,128,32,96,5,64);

  //PRINT_CONFIG(64,64,32,96,5,64);
  //PERF_CONVOLUTION_BRIDGE(64,64,32,96,5,64);

  // PRINT_CONFIG(64,64,32,96,5,96);
  // PERF_CONVOLUTION_BRIDGE(64,64,32,96,5,96);

  // PRINT_CONFIG(64,64,32,96,5,96);
  // PERF_CONVOLUTION_BRIDGE(64,64,32,96,5,96);

  // PRINT_CONFIG(64,64,32,96,5,96);
  // PERF_CONVOLUTION_BRIDGE(64,64,32,96,5,96);

  // PRINT_CONFIG(64,64,32,96,5,96);
  // PERF_CONVOLUTION_BRIDGE(64,64,32,96,5,96);

  return 0;
}
