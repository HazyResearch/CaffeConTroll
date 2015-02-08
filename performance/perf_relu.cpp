#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/ReLUBridge.h"
#include "../src/Report.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <cstring>

void PERF_ReLU_BRIDGE() {
    const int mB = 10;
    const int iD = 8;
    const int iR = 50;
    const int iC = 50;
        
    LogicalCube<float, Layout_CRDB>* data1 = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
    LogicalCube<float, Layout_CRDB>* grad1 = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
    
    LogicalCube<float, Layout_CRDB>* data2 = new LogicalCube<float, Layout_CRDB>(iR, iC, iD, mB);
    LogicalCube<float, Layout_CRDB>* grad2 = new LogicalCube<float, Layout_CRDB> (iR, iC, iD, mB);

    Layer<float, Layout_CRDB>* layer1 = new Layer<float, Layout_CRDB>(data1, grad1);
    Layer<float, Layout_CRDB>* layer2 = new Layer<float, Layout_CRDB>(data2, grad2);

    ReLUBridge<float, Layout_CRDB, float, Layout_CRDB>* ReLUBridge_;
    ReLUBridge_ = new ReLUBridge<float, Layout_CRDB, float, Layout_CRDB>(layer1, layer2); 

    for(int i=0;i<iR*iC*iD*mB;i++){
        data1->p_data[i] = 0.1*(rand()%10 - rand()%10);
    }

    ReLUBridge_->forward();
   
    ReLUBridge_->report_forward_history.print();

    for(int i=0;i<iR*iC*iD*mB;i++){
        grad2->p_data[i] = i*0.1;
    }
    
    for(int i=0;i<iR*iC*iD*mB;i++){
        grad1->p_data[i] = 0;
    }

    ReLUBridge_->backward();
    ReLUBridge_->report_backward_updateweight_history.print();
}

int main(int argc, const char * argv[]) {
  
  PERF_ReLU_BRIDGE();

  return 0;
}
