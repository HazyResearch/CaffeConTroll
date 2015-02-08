#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/SoftmaxLossBridge.h"
#include "../src/Report.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

void PERF_SOFTMAX_BRIDGE() {
    const int mB = 10;
    const int iD = 100;
    const int iR = 1;
    const int iC = 1;
        
    LogicalCube<float, Layout_CRDB>* data1 = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
    LogicalCube<float, Layout_CRDB>* grad1 = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
    
    LogicalCube<float, Layout_CRDB>* data2 = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
    LogicalCube<float, Layout_CRDB>* grad2 = new LogicalCube<float, Layout_CRDB> (iR, iC, iD, mB);

    LogicalCube<float, Layout_CRDB>* label = new LogicalCube<float, Layout_CRDB> (iR, iC, 1, mB);

    Layer<float, Layout_CRDB>* layer1 = new Layer<float, Layout_CRDB>(data1, grad1);
    Layer<float, Layout_CRDB>* layer2 = new Layer<float, Layout_CRDB>(data2, grad2);

    SoftmaxLossBridge<float, Layout_CRDB, float, Layout_CRDB>* SoftmaxBridge_;
    SoftmaxBridge_ = new SoftmaxLossBridge<float, Layout_CRDB, float, Layout_CRDB>(layer1, layer2, label); 

    for(int i=0;i<iR*iC*iD*mB;i++){
        data1->p_data[i] = 0.1*(rand()%10 - rand()%10);
    }

    for(int n=0;n<mB;n++){
        label->p_data[n] = rand()%10;
    }

    SoftmaxBridge_->forward();
   
    SoftmaxBridge_->report_forward_history.print();

    SoftmaxBridge_->backward();
    SoftmaxBridge_->report_backward_updateweight_history.print();
}

int main(int argc, const char * argv[]) {
  
  PERF_SOFTMAX_BRIDGE();

  return 0;
}
