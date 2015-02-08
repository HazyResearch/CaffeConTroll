#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/MaxPoolingBridge.h"
#include "../src/Report.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

void PERF_POOLING_BRIDGE() {
    const int mB = 10;
    const int iD = 3;
    const int oD = iD;
    const int iR = 22;
    const int iC = 22;
    const int k = 2;
    const int s = 2;
    const int p = 0;
    const int oR = static_cast<int>(ceil(static_cast<float>(iR + 2*p - k) / s)) + 1;
    const int oC = static_cast<int>(ceil(static_cast<float>(iC + 2*p - k) / s)) + 1;
    
    LogicalCube<float, Layout_CRDB>* data1 = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
    LogicalCube<float, Layout_CRDB>* grad1 = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
    
    LogicalCube<float, Layout_CRDB>* data2 = new LogicalCube<float, Layout_CRDB>(oR, oC, iD, mB);
    LogicalCube<float, Layout_CRDB>* grad2 = new LogicalCube<float, Layout_CRDB> (oR, oC, iD, mB);

    Layer<float, Layout_CRDB>* layer1 = new Layer<float, Layout_CRDB>(data1, grad1);
    Layer<float, Layout_CRDB>* layer2 = new Layer<float, Layout_CRDB>(data2, grad2);

    BridgeConfig * bconfig = new BridgeConfig(k, 0, 0, s);

    MaxPoolingBridge<float, Layout_CRDB, float, Layout_CRDB>* PoolingBridge_;
    PoolingBridge_ = new MaxPoolingBridge<float, Layout_CRDB, float, Layout_CRDB>(layer1, layer2, bconfig); 

    for(int i=0;i<iR*iC*iD*mB;i++){
        data1->p_data[i] = 0.1*(rand()%10);
    }

    PoolingBridge_->forward();
   
    PoolingBridge_->report_forward_history.print();

    for(int i=0;i<oR*oC*oD*mB;i++){
        grad2->p_data[i] = i*0.1;
    }
    
    for(int i=0;i<iR*iC*iD*mB;i++){
        grad1->p_data[i] = 0;
    }

    PoolingBridge_->backward();
    PoolingBridge_->report_backward_updateweight_history.print();
}

int main(int argc, const char * argv[]) {
  
  PERF_POOLING_BRIDGE();

  return 0;
}
