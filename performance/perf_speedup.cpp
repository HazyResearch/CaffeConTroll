#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/ParallelizedConvolutionBridge.h"
#include "../src/bridges/ConvolutionBridge.h"
#include "../src/Report.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

//#define NPARTITION 1
//#define NTHREADPERPARTITION 1

void PRINT_CONFIG(int iR, int iC, int iD, int oD, int k, int mB){
    std::cout << "Npartition = " << NPARTITION << " " << "N_thread_per_partition = " << NTHREADPERPARTITION << std::endl;
    std::cout << "Input = " << iR << " x " << iC <<" ";
    std::cout << " Kernel = " << k << " x " << k << " ";
    std::cout << " Channels: " << "Input: " << iD << " Output: " << oD << " ";
    std::cout << " BatchSize = " << mB << std::endl;
}

void PERF_CONVOLUTION_BRIDGE(int iR, int iC, int iD, int oD, int k, int mB) {
    const int s = 1;
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

    ParallelizedConvolutionBridge<DataType_SFFloat>* ParallelizedConvolutionBridge_ = 
      new ParallelizedConvolutionBridge<DataType_SFFloat>(layer1, layer2, bconfig,NPARTITION,NTHREADPERPARTITION); 

    ParallelizedConvolutionBridge_->forward();
    cout << "Forward Report - " << endl;   

    cout << "Total Report - " << endl;
    ParallelizedConvolutionBridge_->report_forward_last_transfer.print();
    
    ParallelizedConvolutionBridge_->backward();

    cout << "Backward Report - " << endl;

    cout << "Total Report - " << endl;
    ParallelizedConvolutionBridge_->report_backward_updateweight_last_transfer.print();
    cout << endl;
    
    delete layer1; delete layer2; delete bconfig;
    delete data1; delete data2; delete grad1; delete grad2;
}

int main(int argc, const char * argv[]) {
 
  PRINT_CONFIG(64,64,96,256,5,256);
  PERF_CONVOLUTION_BRIDGE(64,64,96,256,5,256);

  return 0;
}
