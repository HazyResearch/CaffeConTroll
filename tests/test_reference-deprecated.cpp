//
//  main.cpp
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include <iostream>

#include "config.h"
#include "LogicalCube.h"
#include "Connector.h"
#include "Kernel.h"
#include "Bridge.h"
#include "Layer.h"
#include "ParallelizedBridge.h"

using namespace std;

/**
 * Input:
 BATCH 0 DEPTH 0
 a1 d1 g1
 b1 e1 h1
 c1 f1 i1
 BATCH 0 DEPTH 1
 a1' d1' g1'
 b1' e1' h1'
 c1' f1' i1'
 BATCH 1 DEPTH 0
 a2 d2 g2
 b2 e2 h2
 c2 f2 i2
 BATCH 1 DEPTH 1
 a2' d2' g2'
 b2' e2' h2'
 c2' f2' i2'
 *
 * Expect output with Kernel size 3x3:
 *
 BATCH 0 DEPTH 0
 a1 d1 b1 e1 a1' d1' b1' e1'
 d1 g1 e1 h1 d1' g1' e1' h1'
 b1 e1 c1 f1 b1' e1' c1' f1'
 e1 h1 f1 i1 e1' h1' f1' i1'
 a2 d2 b2 e2 a2' d2' b2' e2'
 d2 g2 e2 h2 d2' g2' e2' h2'
 b2 e2 c2 f2 b2' e2' c2' f2'
 e2 h2 f2 i2 e2' h2' f2' i2'
 *
 **/
void TEST_LOWERING() {
    const size_t N = 40;
    const size_t K = 15; // size of the kernel/convolution window (K x K)

    LogicalCube<DataType_FPFloat, Layout_CRDB> cube1(N, N, 2, 2);
    LogicalCube<DataType_FPFloat, Layout_CRDB> cube2(K*K*2, (N-K+1)*(N-K+1)*2, 1, 1);
    LogicalCube<DataType_FPFloat, Layout_CRDB> cube3(K*K*2, (N-K+1)*(N-K+1)*2, 1, 1);

    LoweringConfig lconfig;
    lconfig.kernel_size = K;
    lconfig.stride = 1;

    for (size_t ct = 0, val = 0; ct < N*N*2*2; ct++, val++) {
      cube1.p_data[ct] = val;
    }

    cout << "BEFORE LOWERING: " << endl;
    cube1.logical_print();
    cout << "---------------------" << endl;

    Connector<DataType_FPFloat, Layout_CRDB, DataType_FPFloat, Layout_CRDB, LOWERING_TYPE1>
        connector(&cube1, &cube2, &lconfig);
    connector.lower_cube(&cube1, &cube2);
    cout << "NEW LOWERING: " << endl;
    cube2.logical_print();
    connector.report_last_lowering.print();
    connector.report_history.print();

    Connector<DataType_FPFloat, Layout_CRDB, DataType_FPFloat, Layout_CRDB, LOWERING_TYPE1>
        old_connector(&cube1, &cube3, &lconfig);
    old_connector.old_lower_cube(&cube1, &cube3);
    cout << "OLD LOWERING: " << endl;
    cube3.logical_print();
    old_connector.report_last_lowering.print();
    old_connector.report_history.print();

    //connector.lower_cube(&cube1, &cube2);
    //connector.report_last_lowering.print();
    //connector.report_history.print();
}

void TEST_TIMER(){
    LoweringConfig lconfig;
    lconfig.kernel_size = 3;

    LogicalCube<DataType_SFFloat, Layout_CRDB> cube1(64, 64, 96, 12);

    LogicalCube<DataType_SFFloat, Layout_CRDB> cube2(lconfig.kernel_size*lconfig.kernel_size*96,
                        (64-lconfig.kernel_size+1)*(64-lconfig.kernel_size+1)*12, 1, 1);

    Connector<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, LOWERING_TYPE1>
    connector(&cube1, &cube2, &lconfig);

    connector.lower_cube(&cube1, &cube2);

    connector.report_last_lowering.print();
    connector.report_history.print();
    connector.lower_cube(&cube1, &cube2);
    connector.report_last_lowering.print();
    connector.report_history.print();

}

/**
 * A =
 BATCH 0 DEPTH 0
 0 1 2 3 4
 5 6 7 8 9
 *
   B =
 BATCH 0 DEPTH 0
 3.14 1.57 1.04667
 0.785 0.628 0.523333
 0.448571 0.3925 0.348889
 0.314 0.285455 0.261667
 0.241538 0.224286 0.209333
 *
 * Expect output
 *  A*B =
 BATCH 0 DEPTH 0
 3.5903 3.16651 2.84344
 28.2358 18.6677 14.7929
 *
 **/
void TEST_Kernel_GEMM_OpenBlas_ROW(){

    LogicalCube<DataType_SFFloat, Layout_CRDB> cube1(2, 5, 1, 1);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube2(5, 3, 1, 1);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube3(2, 3, 1, 1);

    for(int i=0;i<10;i++){
        cube1.p_data[i] = 1.0*i;
    }
    for(int i=0;i<15;i++){
        cube2.p_data[i] = 3.14/(i+1);
    }

    cube1.logical_print();
    cube2.logical_print();

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_NOTRANS> kernel(&cube1, &cube2, &cube3);

    kernel.compute(&cube1, &cube2, &cube3);

    cube3.logical_print();

    kernel.report_last_lowering.print();
    kernel.report_history.print();

}


void TEST_Kernel_ELEMENTMUL(){

    LogicalCube<DataType_SFFloat, Layout_CRDB> cube1(5, 5, 1, 1);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube2(5, 5, 1, 1);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube3(5, 5, 1, 1);

    for(int i=0;i<25;i++){
        cube1.p_data[i] = 1.0*i;
    }
    for(int i=0;i<25;i++){
        cube2.p_data[i] = 3.14/(i+1);
    }

    cube1.logical_print();
    cube2.logical_print();

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KernelConfig_GEMM_NOTRANS_NOTRANS> kernel(&cube1, &cube2, &cube3);

    kernel.compute(&cube1, &cube2, &cube3);

    cube3.logical_print();

    kernel.report_last_lowering.print();
    kernel.report_history.print();

}

void TEST_Kernel_ELEMENTMUL_TANHGRAD(){

    LogicalCube<DataType_SFFloat, Layout_CRDB> cube1(5, 5, 1, 1);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube2(5, 5, 1, 1);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube3(5, 5, 1, 1);

    for(int i=0;i<25;i++){
        cube1.p_data[i] = 1.0*i;
    }
    for(int i=0;i<25;i++){
        cube2.p_data[i] = 3.14/(i+1);
    }

    cube1.logical_print();
    cube2.logical_print();

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KernelConfig_TANHGRAD_ON_INPUT1> kernel(&cube1, &cube2, &cube3);

    kernel.compute(&cube1, &cube2, &cube3);

    cube3.logical_print();

    kernel.report_last_lowering.print();
    kernel.report_history.print();

}


/*
 INPUT DATA=
 BATCH 0 DEPTH 0
 1 1 1 0 0
 0 0 0 1 1
 0 1 0 0 1
 1 1 1 0 0
 1 1 1 1 0
 BATCH 1 DEPTH 0
 1 1 0 0 1
 1 0 0 1 0
 1 1 1 1 1
 1 0 1 0 1
 1 0 1 0 1
 INPUT MODEL=
 BATCH 0 DEPTH 0
 1 1 0
 0 0 0
 0 0 0
 BATCH 1 DEPTH 0
 1 0 0
 1 0 1
 1 1 0
 BATCH 2 DEPTH 0
 0 1 0
 1 1 1
 1 0 1
 RESULT=
 BATCH 0 DEPTH 0
 2 2 1
 0 0 1
 1 1 0
 BATCH 0 DEPTH 1
 2 1 0
 1 0 1
 2 2 2
 BATCH 0 DEPTH 2
 2 3 2
 2 3 2
 4 4 3
 BATCH 1 DEPTH 0
 4 4 2
 4 3 3
 4 2 4
 BATCH 1 DEPTH 1
 1 3 3
 3 2 3
 6 4 2
 BATCH 1 DEPTH 2
 4 3 3
 5 3 6
 5 2 5
*/
void TEST_CONV(){

    LogicalCube<DataType_SFFloat, Layout_CRDB> data1(5, 5, 1, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(3, 3, 1, 3);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad1(5, 5, 1, 2);

    LogicalCube<DataType_SFFloat, Layout_CRDB> data2(5-3+1, 5-3+1, 3, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel2(0, 0, 3, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad2(5-3+1, 5-3+1, 3, 2);


    for(int i=0;i<5*5*2;i++){
        data1.p_data[i] = rand()%2;
    }
    for(int i=0;i<3*3*3;i++){
        kernel1.p_data[i] = rand()%2;
    }

    Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &kernel1, &grad1);
    Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &kernel2, &grad2);

    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> forward(&layer1, &layer2);

    forward.forward();

    cout << "\nINPUT DATA=" << endl;
    layer1.p_data_cube->logical_print();

    cout << "\nINPUT MODEL=" << endl;
    layer1.p_model_cube->logical_print();

    cout << "\nRESULT=" << endl;
    layer2.p_data_cube->logical_print();

    forward.report_forward_last_transfer.print();
    forward.report_forward_history.print();

}

/**
 INPUT DATA=
 BATCH 0 DEPTH 0
 1 1 1 0 0
 0 0 0 1 1
 0 1 0 0 1
 1 1 1 0 0
 1 1 1 1 0
 BATCH 1 DEPTH 0
 1 1 0 0 1
 1 0 0 1 0
 1 1 1 1 1
 1 0 1 0 1
 1 0 1 0 1
 INPUT MODEL=
 BATCH 0 DEPTH 0
 1 1 0
 0 0 0
 0 0 0
 BATCH 1 DEPTH 0
 1 0 0
 1 0 1
 1 1 0
 BATCH 2 DEPTH 0
 0 1 0
 1 1 1
 1 0 1
 RESULT=
 BATCH 0 DEPTH 0
 0.964028 0.964028 0.761594
 0 0 0.761594
 0.761594 0.761594 0
 BATCH 0 DEPTH 1
 0.964028 0.761594 0
 0.761594 0 0.761594
 0.964028 0.964028 0.964028
 BATCH 0 DEPTH 2
 0.964028 0.995055 0.964028
 0.964028 0.995055 0.964028
 0.999329 0.999329 0.995055
 BATCH 1 DEPTH 0
 0.999329 0.999329 0.964028
 0.999329 0.995055 0.995055
 0.999329 0.964028 0.999329
 BATCH 1 DEPTH 1
 0.761594 0.995055 0.995055
 0.995055 0.964028 0.995055
 0.999988 0.999329 0.964028
 BATCH 1 DEPTH 2
 0.999329 0.995055 0.995055
 0.999909 0.995055 0.999988
 0.999909 0.964028 0.999909
 **/
void TEST_CONV_WITH_TANH(){

    LogicalCube<DataType_SFFloat, Layout_CRDB> data1(5, 5, 1, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(3, 3, 1, 3);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad1(5, 5, 1, 2);

    LogicalCube<DataType_SFFloat, Layout_CRDB> data2(5-3+1, 5-3+1, 3, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel2(0, 0, 3, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad2(5-3+1, 5-3+1, 3, 2);


    for(int i=0;i<5*5*2;i++){
        data1.p_data[i] = rand()%2;
    }
    for(int i=0;i<3*3*3;i++){
        kernel1.p_data[i] = rand()%2;
    }

    Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &kernel1, &grad1);
    Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &kernel2, &grad2);

    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_TANH> forward(&layer1, &layer2);

    forward.forward();

    cout << "\nINPUT DATA=" << endl;
    layer1.p_data_cube->logical_print();

    cout << "\nINPUT MODEL=" << endl;
    layer1.p_model_cube->logical_print();

    cout << "\nRESULT=" << endl;
    layer2.p_data_cube->logical_print();

    forward.report_forward_last_transfer.print();
    forward.report_forward_history.print();

}


void TEST_CONV_NOTTOY(){

    LogicalCube<DataType_SFFloat, Layout_CRDB> data1(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(5, 5, 96, 256);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad1(64, 64, 96, 12);

    LogicalCube<DataType_SFFloat, Layout_CRDB> data2(64-5+1, 64-5+1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel2(1, 1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad2(64-5+1, 64-5+1, 256, 12);

    Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &kernel1, &grad1);
    Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &kernel2, &grad2);

    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> forward(&layer1, &layer2);

    forward.forward();

    cout << "OVERALL PERFORMANCE" << endl;
    forward.report_forward_last_transfer.print();

    cout << "LOWERING PERFORMANCE" << endl;
    forward.p_forward_lower_connector->report_last_lowering.print();

    cout << "GEMM PERFORMANCE" << endl;
    forward.p_forward_gemm_kernel->report_last_lowering.print();

}

/**
 STEPSIZE = 0.1
 OUTPUT DATA=
 BATCH 0 DEPTH 0
 4 3 3
 3 2 4
 4 4 3
 BATCH 0 DEPTH 1
 3 3 3
 5 6 5
 4 5 4
 BATCH 0 DEPTH 2
 2 3 3
 3 5 3
 3 2 3
 BATCH 1 DEPTH 0
 3 4 3
 5 6 5
 5 5 5
 BATCH 1 DEPTH 1
 1 2 1
 2 2 1
 3 1 1
 BATCH 1 DEPTH 2
 1 3 3
 2 3 3
 1 2 1
 GRADIENT=
 BATCH 0 DEPTH 0
 0 0 0
 1 1 1
 0 0 0
 BATCH 0 DEPTH 1
 0 1 0
 1 0 1
 1 1 1
 BATCH 0 DEPTH 2
 0 0 0
 0 1 0
 0 1 1
 BATCH 1 DEPTH 0
 0 1 0
 1 0 1
 1 0 1
 BATCH 1 DEPTH 1
 0 0 0
 1 1 1
 1 0 0
 BATCH 1 DEPTH 2
 1 1 1
 1 0 0
 0 0 1
 OLD WEIGHT=
 BATCH 0 DEPTH 0
 1 1 1
 0 1 0
 1 1 0
 BATCH 1 DEPTH 0
 0 1 1
 1 1 1
 0 0 1
 BATCH 2 DEPTH 0
 0 0 0
 1 0 0
 1 0 1
 WARNING: You are using the most general version of the lowering function. This might be slow!
 NEW WEIGHT=
 BATCH 0 DEPTH 0
 8.8 13 9.9
 9.7 14.3 11.3
 9.8 9.6 7.4
 BATCH 1 DEPTH 0
 5.1 13.8 10.9
 12.1 12.9 15.6
 7.4 9 9.7
 BATCH 2 DEPTH 0
 0.8 1.4 0.9
 4 2.2 1.4
 4 2.7 4
 NEW GRAD=
 BATCH 0 DEPTH 0
 0 0 0 0 0
 -8 -11 -50 -42 -15
 -3 -35 -30 -50 -8
 -19 -17 -32 -53 -8
 -8 0 -8 -3 -8
 BATCH 1 DEPTH 0
 0 -8 -23 -23 0
 -24 -71 -103 -63 -48
 -42 -127 -142 -134 -71
 -51 -63 -123 -63 -48
 -15 -39 -63 -15 -24
 **/
void TEST_BACKPROP(){

    LogicalCube<DataType_SFFloat, Layout_CRDB> data1(5, 5, 1, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(3, 3, 1, 3);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad1(5, 5, 1, 2);

    LogicalCube<DataType_SFFloat, Layout_CRDB> data2(5-3+1, 5-3+1, 3, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel2(0, 0, 3, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad2(5-3+1, 5-3+1, 3, 2);


    for(int i=0;i<5*5*2;i++){
        data1.p_data[i] = rand()%2;
        grad1.p_data[i] = rand()%2;
    }
    for(int i=0;i<3*3*3;i++){
        kernel1.p_data[i] = rand()%2;
    }
    for(int i=0;i<3*3*3*2;i++){
        data2.p_data[i] = rand()%2;
        grad2.p_data[i] = rand()%2;
    }

    Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &kernel1, &grad1);
    Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &kernel2, &grad2);

    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> forward(&layer1, &layer2);

    forward.forward();

    cout << "STEPSIZE = " << forward.stepsize << endl;

    cout << "\nOUTPUT DATA=" << endl;
    layer2.p_data_cube->logical_print();

    cout << "\nGRADIENT=" << endl;
    layer2.p_gradient_cube->logical_print();

    cout << "\nOLD WEIGHT=" << endl;
    layer1.p_model_cube->logical_print();

    forward.backward();

    cout << "\nNEW WEIGHT=" << endl;
    layer1.p_model_cube->logical_print();

    cout << "\nNEW GRAD=" << endl;
    layer1.p_gradient_cube->logical_print();

    forward.report_forward_last_transfer.print();
    forward.report_forward_history.print();

}


void TEST_BACKPROP_NOTTOY(){

    LogicalCube<DataType_SFFloat, Layout_CRDB> data1(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(5, 5, 96, 256);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad1(64, 64, 96, 12);

    LogicalCube<DataType_SFFloat, Layout_CRDB> data2(64-5+1, 64-5+1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel2(0, 0, 256, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad2(64-5+1, 64-5+1, 256, 12);

    Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &kernel1, &grad1);
    Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &kernel2, &grad2);

    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> forward(&layer1, &layer2);

    forward.backward();

    cout << "OVERALL PERFORMANCE:" << endl;
    forward.report_backward_updateweight_last_transfer.print();

    cout << "Propogate Gradient ELEMENT-WISE MUL:" << endl;
    forward.p_backward_element_mul_kernel->report_last_lowering.print();

    cout << "Upgrade Kernel MUL:" << endl;
    forward.p_backward_gemm_updateweight_kernel->report_last_lowering.print();

    cout << "Upgrade Grad MUL:" << endl;
    forward.p_backward_gemm_updategrad_kernel->report_last_lowering.print();

    cout << "INVERSE LOWERING:" << endl;
    forward.p_forward_lower_connector->report_last_inverse_lowering.print();


}


void TEST_PHYSICAL_EXECUTOR(){

    LogicalCube<DataType_SFFloat, Layout_CRDB> data11(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> data12(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> data13(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> data14(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> data15(64, 64, 96, 12);

    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(5, 5, 96, 256);

    LogicalCube<DataType_SFFloat, Layout_CRDB> grad11(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad12(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad13(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad14(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad15(64, 64, 96, 12);

    LogicalCube<DataType_SFFloat, Layout_CRDB> data21(64-5+1, 64-5+1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> data22(64-5+1, 64-5+1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> data23(64-5+1, 64-5+1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> data24(64-5+1, 64-5+1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> data25(64-5+1, 64-5+1, 256, 12);

    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel2(0, 0, 256, 2);

    LogicalCube<DataType_SFFloat, Layout_CRDB> grad21(64-5+1, 64-5+1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad22(64-5+1, 64-5+1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad23(64-5+1, 64-5+1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad24(64-5+1, 64-5+1, 256, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad25(64-5+1, 64-5+1, 256, 12);

    Layer<DataType_SFFloat, Layout_CRDB> layer11(&data11, &kernel1, &grad11);
    Layer<DataType_SFFloat, Layout_CRDB> layer12(&data12, &kernel1, &grad12);
    Layer<DataType_SFFloat, Layout_CRDB> layer13(&data13, &kernel1, &grad13);
    Layer<DataType_SFFloat, Layout_CRDB> layer14(&data14, &kernel1, &grad14);
    Layer<DataType_SFFloat, Layout_CRDB> layer15(&data15, &kernel1, &grad15);

    Layer<DataType_SFFloat, Layout_CRDB> layer21(&data21, &kernel2, &grad21);
    Layer<DataType_SFFloat, Layout_CRDB> layer22(&data22, &kernel2, &grad22);
    Layer<DataType_SFFloat, Layout_CRDB> layer23(&data23, &kernel2, &grad23);
    Layer<DataType_SFFloat, Layout_CRDB> layer24(&data24, &kernel2, &grad24);
    Layer<DataType_SFFloat, Layout_CRDB> layer25(&data25, &kernel2, &grad25);

    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> bridge1(&layer11, &layer21);
    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> bridge2(&layer12, &layer22);
    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> bridge3(&layer13, &layer23);
    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> bridge4(&layer14, &layer24);
    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> bridge5(&layer15, &layer25);

    bridge1.run_with_n_threads = 1;
    bridge2.run_with_n_threads = 1;
    bridge3.run_with_n_threads = 1;
    bridge4.run_with_n_threads = 1;
    bridge5.run_with_n_threads = 1;

    PhysicalStratum stratum;
    stratum.executors.push_back((PhysicalOperator*)&bridge1);
    stratum.executors.push_back((PhysicalOperator*)&bridge2);
    stratum.executors.push_back((PhysicalOperator*)&bridge3);
    stratum.executors.push_back((PhysicalOperator*)&bridge4);
    stratum.executors.push_back((PhysicalOperator*)&bridge5);

    //stratum.forward();

    //stratum.report_forward_last_transfer.print();

    stratum.backward();

    stratum.report_backward_updateweight_last_transfer.print();

}

void TEST_PHYSICAL_PARALLELBRIDGE_WITH_GROUND_TRUTH(){

    LogicalCube<DataType_SFFloat, Layout_CRDB> data1(5, 5, 1, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(3, 3, 1, 3);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad1(5, 5, 1, 2);

    LogicalCube<DataType_SFFloat, Layout_CRDB> data2(5-3+1, 5-3+1, 3, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> kernel2(0, 0, 3, 2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> grad2(5-3+1, 5-3+1, 3, 2);


    for(int i=0;i<5*5*2;i++){
        data1.p_data[i] = rand()%2;
    }
    for(int i=0;i<3*3*3;i++){
        kernel1.p_data[i] = rand()%2;
    }

    Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &kernel1, &grad1);
    Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &kernel2, &grad2);

    ParallelizedBridge<DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> parallel_bridge(&layer1, &layer2, 2, 1);

    parallel_bridge.forward();
    parallel_bridge.report_forward_last_transfer.print();

    data2.logical_print();

}


void TEST_PHYSICAL_PARALLELBRIDGE(){

    Layer<DataType_SFFloat, Layout_CRDB> * layer1 = Layer<DataType_SFFloat, Layout_CRDB>::make_layer(64, 96, 128, 5, 256);

    Layer<DataType_SFFloat, Layout_CRDB> * layer2 = Layer<DataType_SFFloat, Layout_CRDB>::make_layer(64-5+1, 256, 128, 2, 2);

    ParallelizedBridge<DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> parallel_bridge(layer1, layer2, 4, 1);

    parallel_bridge.forward();

    parallel_bridge.backward();

    parallel_bridge.report_forward_last_transfer.print();

    parallel_bridge.report_backward_updateweight_last_transfer.print();
}


int main(int argc, const char * argv[]) {

    //TEST_PHYSICAL_PARALLELBRIDGE();

    //TEST_PHYSICAL_PARALLELBRIDGE_WITH_GROUND_TRUTH();

    //TEST_PHYSICAL_PARALLELBRIDGE();

    //TEST_PHYSICAL_EXECUTOR();

    //TEST_CONV_WITH_TANH();

    //TEST_BACKPROP_NOTTOY();

    //TEST_Kernel_ELEMENTMUL_TANHGRAD();

    //TEST_Kernel_ELEMENTMUL();

    //TEST_CONV_NOTTOY();

    //TEST_CONV();

    TEST_LOWERING();

    //TEST_TIMER();

    //TEST_Kernel_GEMM_OpenBlas_ROW();

    /*
    LoweringConfig lconfig;
    lconfig.kernel_size = 3;
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube_kernel(3, 3, 96, 64);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube_output(62, 62, 64, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube1(64, 64, 96, 12);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube2(lconfig.kernel_size*lconfig.kernel_size*96,
                                              (64-lconfig.kernel_size+1)*(64-lconfig.kernel_size+1)*12, 1, 1);
    Connector<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Connector_Lowering_R1C1>
    connector(&cube1, &cube2, &lconfig);
    connector.transfer(&cube1, &cube2);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube_kernel_matrix(cube_kernel.p_data, 64, 3*3*96, 1, 1);
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube_output_matrix(cube_output.p_data, 64, (64-lconfig.kernel_size+1)*(64-lconfig.kernel_size+1)*12, 1, 1);
    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Kernel_GEMM_OpenBlas> kernel(&cube_kernel_matrix, &cube2, &cube_output_matrix);
    kernel.compute(&cube_kernel_matrix, &cube2, &cube_output_matrix);
    kernel.report_last_lowering.print();
    kernel.report_history.print();
    */

    /*
    connector.report_last_lowering.print();
    connector.report_history.print();
    connector.transfer(&cube1, &cube2);
    connector.report_last_lowering.print();
    connector.report_history.print();
     */


    /*
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube1(
        10, 10, 10, 10
    );
    LogicalCube<DataType_SFFloat, Layout_CRDB> cube2(
        3*3*10, (10-3+1)*(10-3+1)*10, 1, 1
    );
    */

    //DataType_SFFloat * data = cube.logical_get(1,2,3,4);
    //*data = 5;
    //data = cube.logical_get(1,2,3,4);
    //cout << *data << endl;
    //cube.physical_get_RCslice(3,4);

    /*
    LoweringConfig lconfig;
    lconfig.kernel_size = 3;
    Connector<DataType_SFFloat, Layout_CRDB,
              DataType_SFFloat, Layout_CRDB,
              Connector_Lowering_R1C1>connector(&cube1, &cube2, &lconfig);
    connector.transfer(&cube1, &cube2);
    */

    return 0;
}