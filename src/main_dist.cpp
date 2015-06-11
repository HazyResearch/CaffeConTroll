//
//  main.cpp
//  CCT_Distributed
//
//  Created by Igor B on 6/10/15.
//  Copyright (c) 2015 Igor B. All rights reserved.
//

#include <iostream>
#include "/usr/local/include/mpi.h"
#include "LogicalCube.h"
#include "sched/DeviceDriver_CPU.cpp"
#include "sched/DeviceDriver.cpp"
#include "bridges/ConvolutionBridge.h"
//#include "bridges/ParallelizedBridge.h"
#include "Layer.h"
#include "parser/parser.h"
#include "parser/cnn.pb.h"
#include "sched/DeviceHeader.h"
#include "timer.h"
#include "Report.h"
//#include "kernels/lowering.h"
//#include "cube_funcs.h"
//#include "AsyncBridge.h"
#define SIZE 1024
#define myLog(a) std::cout <<__FILE__<<": "<< __PRETTY_FUNCTION__<<": "<<__LINE__ <<": "<<a<<"\n"
#include "Container.h"

int main(int argc, char** argv) {
    int rank,numtasks;
    //MPI_Status status;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    /*
     int mB = 1;//4;
     int iD = 1; //3;
     int iR = 3;//20;
     int iC = 3;//20;
     int oD = 1;//10 //output depth
     int k = 3;//5; //kernel size
     int s = 1;//4; //stride
     int p = 1;//2; //padding
     */
    
    int mB = strtol(argv[1], NULL, 10);//4;
    int iD = strtol(argv[2], NULL, 10); //3;
    int iR = strtol(argv[3], NULL, 10);//20;
    int iC = strtol(argv[4], NULL, 10);//20;
    int oD = strtol(argv[5], NULL, 10);//10 //output depth
    int k = strtol(argv[6], NULL, 10);//5; //kernel size
    int s = strtol(argv[7], NULL, 10);//4; //stride
    int p = strtol(argv[8], NULL, 10);//2; //padding
    
    cnn::LayerParameter * p_layer_param = new cnn::LayerParameter();
    p_layer_param->set_type(cnn::LayerParameter_LayerType_CONVOLUTION);
    p_layer_param->set_gpu_0_batch_proportion(0); //igor
    p_layer_param->mutable_convolution_param()->set_num_output(oD);
    p_layer_param->mutable_convolution_param()->set_kernel_size(k);
    p_layer_param->mutable_convolution_param()->set_pad(p);
    p_layer_param->mutable_convolution_param()->set_stride(s);
    
    
    Container container(rank,mB,iD,iR, iC, p_layer_param);
    //container.feedModel();
    container.run();
    //container.run();
    std::cout<<"===Done!==\n";
}
