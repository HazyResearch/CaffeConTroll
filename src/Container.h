//
//  Container.h
//  xCCT
//
//  Created by Igor B on 5/15/15.
//  Copyright (c) 2015 Igor B. All rights reserved.
//

#ifndef __xCCT__Container__
#define __xCCT__Container__

#include <stdio.h>
#include "LogicalCube.h"
#include "parser/cnn.pb.h"
#include "bridges/ConvolutionBridge.h"
#include "bridges/FullyConnectedBridge.h"
#include "/usr/local/include/mpi.h"
#include "algorithms/GradientUpdater.h"

#define MSG_SIZE 6
enum requestType {EVICT,FEED,SYNC};
enum dataClass {LABEL,FEATURES,GRADS,MODEL,BIAS};
enum other {ANY_BATCH=-1,SCHED=0,METADATA};


class Container{
public:
    //Container(int myRank,int _mB,int _iD,int _iR,int _iC,int _oD,int _k,int _s,int _p);
    Container(int _myRank,int _mB,int _iD,int _iR,int _iC, cnn::LayerParameter * _p_layer_param);
    void run();
    ~Container();

    int feedModel();
    int feed(LogicalCube<float, Layout_CRDB>* p_input, int batch_id,int dataType);
    int evict(LogicalCube<float, Layout_CRDB>* p_input, int batch_id,int dataType);
    
    
private:
    int myRank,mB,iD,iR,iC,oD,oR,oC, k,s,p;
    int requestID=0;
    LogicalCube<float, Layout_CRDB> * p_model;
    LogicalCube<float, Layout_CRDB> * p_bias;
    LogicalCube<float, Layout_CRDB> * p_X;
    LogicalCube<float, Layout_CRDB> * p_Y;
    LogicalCube<float, Layout_CRDB> * p_dX;
    LogicalCube<float, Layout_CRDB> * p_dY;
    Layer<float, Layout_CRDB> * p_input_layer;
    Layer<float, Layout_CRDB> * p_output_layer;
    
	cnn::LayerParameter * p_layer_param;
    cnn::ConvolutionParameter  * p_conv_param;
    cnn::SolverParameter * p_solver_param;
    SGDGradientUpdater<float, CPUDriver> * p_grad_updater;
    SGDGradientUpdater<float, CPUDriver> * p_grad_updater_bias;
    ConvolutionBridge<float, Layout_CRDB, float, Layout_CRDB, CPUDriver>  * p_bridge;
    //AbstractBridge<float, Layout_CRDB, float, Layout_CRDB, CPUDriver>  * p_bridge;
    
    void receiveCube(LogicalCube<float, Layout_CRDB>* p_input, int src_rank);
    void sendCube(LogicalCube<float, Layout_CRDB>* p_output, int dst_rank);
    

    
};

static inline size_t compute_conv_next_layer_dimension(const size_t R_i, const size_t K,
                                                       const size_t padding, const size_t stride ) {
    return (R_i + 2 * padding - K) / stride + 1;
}

#include "Container.cpp"
#endif /* defined(__xCCT__Container__) */
