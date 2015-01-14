//
//  main.cpp
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include <iostream>

#include "config.h"
#include "Cube.h"
#include "Connector.h"
#include "Kernel.h"
#include "Bridge.h"
#include "Layer.h"

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
void TEST_LOWERING(){
    Cube<DataType_String, Layout_CRDB> cube1(3, 3, 2, 2);
    
    Cube<DataType_String, Layout_CRDB> cube2(2*2*2, (3-2+1)*(3-2+1)*2, 1, 1);
    
    LoweringConfig lconfig;
    lconfig.kernel_size = 2;
    
    Connector<DataType_String, Layout_CRDB, DataType_String, Layout_CRDB, Connector_Lowering_R1C1>
        connector(&cube1, &cube2, &lconfig);
    
    size_t ct = 0;
    cube1.p_data[ct++] = "a1"; cube1.p_data[ct++] = "b1";
    cube1.p_data[ct++] = "c1"; cube1.p_data[ct++] = "d1";
    cube1.p_data[ct++] = "e1"; cube1.p_data[ct++] = "f1";
    cube1.p_data[ct++] = "g1"; cube1.p_data[ct++] = "h1";
    cube1.p_data[ct++] = "i1";
    
    cube1.p_data[ct++] = "a2"; cube1.p_data[ct++] = "b2";
    cube1.p_data[ct++] = "c2"; cube1.p_data[ct++] = "d2";
    cube1.p_data[ct++] = "e2"; cube1.p_data[ct++] = "f2";
    cube1.p_data[ct++] = "g2"; cube1.p_data[ct++] = "h2";
    cube1.p_data[ct++] = "i2";
    
    cube1.p_data[ct++] = "a1'"; cube1.p_data[ct++] = "b1'";
    cube1.p_data[ct++] = "c1'"; cube1.p_data[ct++] = "d1'";
    cube1.p_data[ct++] = "e1'"; cube1.p_data[ct++] = "f1'";
    cube1.p_data[ct++] = "g1'"; cube1.p_data[ct++] = "h1'";
    cube1.p_data[ct++] = "i1'";
    
    cube1.p_data[ct++] = "a2'"; cube1.p_data[ct++] = "b2'";
    cube1.p_data[ct++] = "c2'"; cube1.p_data[ct++] = "d2'";
    cube1.p_data[ct++] = "e2'"; cube1.p_data[ct++] = "f2'";
    cube1.p_data[ct++] = "g2'"; cube1.p_data[ct++] = "h2'";
    cube1.p_data[ct++] = "i2'";
    
    connector.transfer(&cube1, &cube2);
    
    cube2.logical_print();
    
    connector.report_last_transfer.print();
    connector.report_history.print();
    connector.transfer(&cube1, &cube2);
    connector.report_last_transfer.print();
    connector.report_history.print();
}

void TEST_TIMER(){
    LoweringConfig lconfig;
    lconfig.kernel_size = 3;
    
    Cube<DataType_SFFloat, Layout_CRDB> cube1(64, 64, 96, 12);
    
    Cube<DataType_SFFloat, Layout_CRDB> cube2(lconfig.kernel_size*lconfig.kernel_size*96,
                        (64-lconfig.kernel_size+1)*(64-lconfig.kernel_size+1)*12, 1, 1);
    
    Connector<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Connector_Lowering_R1C1>
    connector(&cube1, &cube2, &lconfig);
    
    connector.transfer(&cube1, &cube2);
    
    connector.report_last_transfer.print();
    connector.report_history.print();
    connector.transfer(&cube1, &cube2);
    connector.report_last_transfer.print();
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
    
    Cube<DataType_SFFloat, Layout_CRDB> cube1(2, 5, 1, 1);
    Cube<DataType_SFFloat, Layout_CRDB> cube2(5, 3, 1, 1);
    Cube<DataType_SFFloat, Layout_CRDB> cube3(2, 3, 1, 1);

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
    
    kernel.report_last_transfer.print();
    kernel.report_history.print();

}


void TEST_Kernel_ELEMENTMUL(){
    
    Cube<DataType_SFFloat, Layout_CRDB> cube1(5, 5, 1, 1);
    Cube<DataType_SFFloat, Layout_CRDB> cube2(5, 5, 1, 1);
    Cube<DataType_SFFloat, Layout_CRDB> cube3(5, 5, 1, 1);
    
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
    
    kernel.report_last_transfer.print();
    kernel.report_history.print();
    
}

void TEST_Kernel_ELEMENTMUL_TANHGRAD(){
    
    Cube<DataType_SFFloat, Layout_CRDB> cube1(5, 5, 1, 1);
    Cube<DataType_SFFloat, Layout_CRDB> cube2(5, 5, 1, 1);
    Cube<DataType_SFFloat, Layout_CRDB> cube3(5, 5, 1, 1);
    
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
    
    kernel.report_last_transfer.print();
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
 
    Cube<DataType_SFFloat, Layout_CRDB> data1(5, 5, 1, 2);
    Cube<DataType_SFFloat, Layout_CRDB> kernel1(3, 3, 1, 3);
    Cube<DataType_SFFloat, Layout_CRDB> grad1(5, 5, 1, 2);
    
    Cube<DataType_SFFloat, Layout_CRDB> data2(5-3+1, 5-3+1, 3, 2);
    Cube<DataType_SFFloat, Layout_CRDB> kernel2(0, 0, 3, 2);
    Cube<DataType_SFFloat, Layout_CRDB> grad2(5-3+1, 5-3+1, 3, 2);

    
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
    
    std::cout << "\nINPUT DATA=" << std::endl;
    layer1.p_data_cube->logical_print();
    
    std::cout << "\nINPUT MODEL=" << std::endl;
    layer1.p_model_cube->logical_print();
    
    std::cout << "\nRESULT=" << std::endl;
    layer2.p_data_cube->logical_print();
    
    forward.report_forward_last_transfer.print();
    forward.report_forward_history.print();

}


void TEST_CONV_NOTTOY(){
    
    Cube<DataType_SFFloat, Layout_CRDB> data1(64, 64, 96, 12);
    Cube<DataType_SFFloat, Layout_CRDB> kernel1(5, 5, 96, 256);
    Cube<DataType_SFFloat, Layout_CRDB> grad1(64, 64, 96, 12);
    
    Cube<DataType_SFFloat, Layout_CRDB> data2(64-5+1, 64-5+1, 256, 12);
    Cube<DataType_SFFloat, Layout_CRDB> kernel2(1, 1, 256, 12);
    Cube<DataType_SFFloat, Layout_CRDB> grad2(64-5+1, 64-5+1, 256, 12);
    
    Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &kernel1, &grad1);
    Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &kernel2, &grad2);
    
    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> forward(&layer1, &layer2);
    
    forward.forward();
    
    std::cout << "OVERALL PERFORMANCE" << std::endl;
    forward.report_forward_last_transfer.print();
    
    std::cout << "LOWERING PERFORMANCE" << std::endl;
    forward.p_forward_lower_connector->report_last_transfer.print();
    
    std::cout << "GEMM PERFORMANCE" << std::endl;
    forward.p_forward_gemm_kernel->report_last_transfer.print();
    
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
    
    Cube<DataType_SFFloat, Layout_CRDB> data1(5, 5, 1, 2);
    Cube<DataType_SFFloat, Layout_CRDB> kernel1(3, 3, 1, 3);
    Cube<DataType_SFFloat, Layout_CRDB> grad1(5, 5, 1, 2);
    
    Cube<DataType_SFFloat, Layout_CRDB> data2(5-3+1, 5-3+1, 3, 2);
    Cube<DataType_SFFloat, Layout_CRDB> kernel2(0, 0, 3, 2);
    Cube<DataType_SFFloat, Layout_CRDB> grad2(5-3+1, 5-3+1, 3, 2);
    
    
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
    
    std::cout << "STEPSIZE = " << forward.stepsize << std::endl;
    
    std::cout << "\nOUTPUT DATA=" << std::endl;
    layer2.p_data_cube->logical_print();
    
    std::cout << "\nGRADIENT=" << std::endl;
    layer2.p_gradient_cube->logical_print();

    std::cout << "\nOLD WEIGHT=" << std::endl;
    layer1.p_model_cube->logical_print();
    
    forward.backward();
    
    std::cout << "\nNEW WEIGHT=" << std::endl;
    layer1.p_model_cube->logical_print();
    
    std::cout << "\nNEW GRAD=" << std::endl;
    layer1.p_gradient_cube->logical_print();
    
    forward.report_forward_last_transfer.print();
    forward.report_forward_history.print();
    
}


void TEST_BACKPROP_NOTTOY(){
    
    Cube<DataType_SFFloat, Layout_CRDB> data1(64, 64, 96, 12);
    Cube<DataType_SFFloat, Layout_CRDB> kernel1(5, 5, 96, 256);
    Cube<DataType_SFFloat, Layout_CRDB> grad1(64, 64, 96, 12);
    
    Cube<DataType_SFFloat, Layout_CRDB> data2(64-5+1, 64-5+1, 256, 12);
    Cube<DataType_SFFloat, Layout_CRDB> kernel2(0, 0, 256, 2);
    Cube<DataType_SFFloat, Layout_CRDB> grad2(64-5+1, 64-5+1, 256, 12);
    
    Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &kernel1, &grad1);
    Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &kernel2, &grad2);
    
    Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> forward(&layer1, &layer2);
    
    forward.backward();
    
    std::cout << "OVERALL PERFORMANCE:" << std::endl;
    forward.report_backward_updateweight_last_transfer.print();
    
    std::cout << "Propogate Gradient ELEMENT-WISE MUL:" << std::endl;
    forward.p_backward_element_mul_kernel->report_last_transfer.print();

    std::cout << "Upgrade Kernel MUL:" << std::endl;
    forward.p_backward_gemm_updateweight_kernel->report_last_transfer.print();

    std::cout << "Upgrade Grad MUL:" << std::endl;
    forward.p_backward_gemm_updategrad_kernel->report_last_transfer.print();

    std::cout << "INVERSE LOWERING:" << std::endl;
    forward.p_forward_lower_connector->report_last_inverse_transfer.print();

    
}



int main(int argc, const char * argv[]) {
    
    TEST_BACKPROP_NOTTOY();
    
    //TEST_Kernel_ELEMENTMUL_TANHGRAD();
    
    //TEST_Kernel_ELEMENTMUL();
    
    //TEST_CONV_NOTTOY();
    
    //TEST_CONV();
    
    //TEST_LOWERING();

    //TEST_TIMER();
    
    //TEST_Kernel_GEMM_OpenBlas_ROW();
    
    /*
    LoweringConfig lconfig;
    lconfig.kernel_size = 3;
    
    Cube<DataType_SFFloat, Layout_CRDB> cube_kernel(3, 3, 96, 64);
 
    Cube<DataType_SFFloat, Layout_CRDB> cube_output(62, 62, 64, 12);
    
    Cube<DataType_SFFloat, Layout_CRDB> cube1(64, 64, 96, 12);
    
    Cube<DataType_SFFloat, Layout_CRDB> cube2(lconfig.kernel_size*lconfig.kernel_size*96,
                                              (64-lconfig.kernel_size+1)*(64-lconfig.kernel_size+1)*12, 1, 1);
    
    Connector<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Connector_Lowering_R1C1>
    connector(&cube1, &cube2, &lconfig);
    
    connector.transfer(&cube1, &cube2);
    
    Cube<DataType_SFFloat, Layout_CRDB> cube_kernel_matrix(cube_kernel.p_data, 64, 3*3*96, 1, 1);
    Cube<DataType_SFFloat, Layout_CRDB> cube_output_matrix(cube_output.p_data, 64, (64-lconfig.kernel_size+1)*(64-lconfig.kernel_size+1)*12, 1, 1);
    
    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Kernel_GEMM_OpenBlas> kernel(&cube_kernel_matrix, &cube2, &cube_output_matrix);

    kernel.compute(&cube_kernel_matrix, &cube2, &cube_output_matrix);
    
    kernel.report_last_transfer.print();
    kernel.report_history.print();
    */
    
    /*
    connector.report_last_transfer.print();
    connector.report_history.print();
    connector.transfer(&cube1, &cube2);
    connector.report_last_transfer.print();
    connector.report_history.print();
     */
    
    
    /*
    Cube<DataType_SFFloat, Layout_CRDB> cube1(
        10, 10, 10, 10
    );
    
    Cube<DataType_SFFloat, Layout_CRDB> cube2(
        3*3*10, (10-3+1)*(10-3+1)*10, 1, 1
    );
    */
    
    //DataType_SFFloat * data = cube.logical_get(1,2,3,4);
    //*data = 5;
    //data = cube.logical_get(1,2,3,4);
    //std::cout << *data << std::endl;
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
