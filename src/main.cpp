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
#include "bridges/MaxPoolingBridge.h"
#include "bridges/ReLUBridge.h"
#include "bridges/ConvolutionBridge.h"
#include "bridges/SoftmaxLossBridge.h"
#include "Layer.h"
//#include "bridges/ParallelizedBridge.h"
#include "parser/parser.h"

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
  const size_t N = 3;
  const size_t K = 2; // size of the kernel/convolution window (K x K)
  const size_t D = 2;
  const size_t B = 3;

  LogicalCube<DataType_FPFloat, Layout_CRDB> cube1(N, N, D, B);
  LogicalCube<DataType_FPFloat, Layout_CRDB> cube2(K*K*D, (N-K+1)*(N-K+1)*B, 1, 1);
  cube2.reset_cube();

  BridgeConfig bconfig;
  bconfig.kernel_size = K;
  bconfig.stride = 1;

  for (size_t ct = 0, val = 0; ct < N*N*D*B; ct++, val++) {
    cube1.p_data[ct] = val;
  }

  cout << "BEFORE LOWERING: " << endl;
  cube1.logical_print();

  cout << "---------------------" << endl;

  Connector<DataType_FPFloat, Layout_CRDB, DataType_FPFloat, Layout_CRDB, LOWERING_TYPE1>
    connector(&cube1, &cube2, &bconfig);
  connector.lower_cube(&cube1, &cube2);

  cout << "AFTER LOWERING: " << endl;
  cube2.logical_print();

  connector.report_last_lowering.print();
  connector.report_history.print();
}

template <typename T>
void simple_conv(LogicalCube<T, Layout_CRDB>* in, LogicalCube<T, Layout_CRDB>* kernel, LogicalCube<T, Layout_CRDB>* out) {
  int ofm = out->D;
  int ifm = in->D;
  for (int n = 0; n < out->B; n++) {
    for (int o = 0; o < ofm; o++) {
      for (int k = 0; k < ifm; k++) {
        for (int y = 0; y < out->R; y++) {
          for (int x = 0; x < out->C; x++) {
            for (int p = 0; p < kernel->R; p++) {
              for (int q = 0; q < kernel->C; q++) {
                int in_y = y + p;
                int in_x = x + q;
                *out->logical_get(y, x, o, n) +=
                  *in->logical_get(in_y, in_x, k, n)*
                  *kernel->logical_get(p, q, k, o);
              }
            }
          }
        }
      }
    }
  }
}

void TEST_CONVOLUTION_BRIDGE() {
  const size_t N = 5;
  const size_t K = 3;
  const size_t D = 3;
  const size_t B = 2;
  const size_t O = 3;

  LogicalCube<DataType_SFFloat, Layout_CRDB> data1(N, N, D, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(K, K, D, O);
  LogicalCube<DataType_SFFloat, Layout_CRDB> bias(1, 1, O, 1);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad1(N, N, D, B);

  LogicalCube<DataType_SFFloat, Layout_CRDB> data2(N-K+1, N-K+1, O, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad2(N-K+1, N-K+1, O, B);

  xavier_initialize(data1.p_data, N*N*D*B, B);
  constant_initialize<float>(bias.p_data, 0.0, O);

  kernel1.reset_cube();
  for (int i=0;i<K*K*D*O;i++) {
    kernel1.p_data[i] = rand() % 2;
  }

  Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &grad1);
  Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &grad2);

  ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> forward(&layer1, &layer2, &kernel1, &bias);

  cout << "\nINPUT DATA=" << endl;
  layer1.p_data_cube->logical_print();

  cout << "\nINPUT MODEL=" << endl;
  kernel1.logical_print();

  cout << "\nINPUT BIAS=" << endl;
  bias.logical_print();

  forward.forward();

  LogicalCube<DataType_SFFloat, Layout_CRDB> out_expected(N-K+1, N-K+1, O, B);
  out_expected.reset_cube();
  simple_conv<DataType_SFFloat>(&data1, &kernel1, &out_expected);

  cout << "\nRESULT=" << endl;
  layer2.p_data_cube->logical_print();

  cout << "\nEXPECTED RESULT=" << endl;
  out_expected.logical_print();
  forward.report_forward_last_transfer.print();
  forward.report_forward_history.print();
}

void TEST_SOFTMAX() {
  static const int mB = 10;
  static const int iD = 15;

  LogicalCube<DataType_SFFloat, Layout_CRDB> data1(1, 1, iD, mB);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad1(1, 1, iD, mB);

  LogicalCube<DataType_SFFloat, Layout_CRDB> data2(1, 1, 1, mB);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad2(1, 1, 1, mB);

  LogicalCube<DataType_SFFloat, Layout_CRDB> label(1, 1, 1, mB);

  Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &grad1);
  Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &grad2);

  SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> softmaxBridge_(&layer1, &layer2, &label);

  srand(1);
  for(int i=0; i < iD*mB; i++){
    data1.p_data[i] = (rand()%5)*0.1;
  }
  cout << "INPUT DATA:" << endl;
  data1.logical_print();

  srand(0);
  for(int n=0;n<mB;n++){
    label.p_data[n] = rand()%10;
  }
  cout << "LABEL DATA:" << endl;
  label.logical_print();

  softmaxBridge_.forward();
  cout << "LOSS: " << softmaxBridge_.loss << endl;
}

int main(int argc, const char * argv[]) {
  //TEST_LOWERING();

  //TEST_CONVOLUTION_BRIDGE();

  TEST_SOFTMAX();

  return 0;
}
