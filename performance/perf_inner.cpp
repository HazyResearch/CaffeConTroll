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

void PERF_INNER_BRIDGE() {
    const int mB = 10;
    const int iD = 3;
    const int oD = 5;
    const int iR = 22;
    const int iC = 22;
    const int k = iR;
    const int s = 1;
    const int p = 0;
    const int oR = 1;
    const int oC = 1;
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
   
    ConvolutionBridge_->report_forward_history.print();

    for(int i=0;i<oR*oC*oD*mB;i++){
        grad2->p_data[i] = i*0.1;
    }
    
    for(int i=0;i<iR*iC*iD*mB;i++){
        grad1->p_data[i] = 0;
    }

    ConvolutionBridge_->backward();
    ConvolutionBridge_->report_backward_updateweight_history.print();
}

int main(int argc, const char * argv[]) {
  
  PERF_INNER_BRIDGE();

  return 0;
}
