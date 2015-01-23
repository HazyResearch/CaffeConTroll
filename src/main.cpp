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
#include "parser/parser.h"
#include "Operation.h"
#include "Corpus.h"

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

  LoweringConfig lconfig;
  lconfig.kernel_size = K;
  lconfig.stride = 1;

  for (size_t ct = 0, val = 0; ct < N*N*D*B; ct++, val++) {
    cube1.p_data[ct] = val;
  }

  cout << "BEFORE LOWERING: " << endl;
  cube1.logical_print();

  cout << "---------------------" << endl;

  Connector<DataType_FPFloat, Layout_CRDB, DataType_FPFloat, Layout_CRDB, LOWERING_TYPE1>
    connector(&cube1, &cube2, &lconfig);
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

void TEST_BRIDGE() {
  const size_t N = 5;
  const size_t K = 3;
  const size_t D = 2;
  const size_t B = 3;
  const size_t O = 2;

  LogicalCube<DataType_SFFloat, Layout_CRDB> data1(N, N, D, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(K, K, D, O);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad1(N, N, D, B);

  LogicalCube<DataType_SFFloat, Layout_CRDB> data2(N-K+1, N-K+1, O, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> kernel2(0, 0, O, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad2(N-K+1, N-K+1, O, B);

  data1.reset_cube();
  kernel1.reset_cube();

  for (int i=0;i<N*N*D*B;i++) {
    data1.p_data[i] = rand() % 2;
  }
  for (int i=0;i<K*K*D*O;i++) {
    kernel1.p_data[i] = rand() % 2;
  }

  Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &kernel1, &grad1);
  Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &kernel2, &grad2);

  Bridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC> forward(&layer1, &layer2);

  cout << "\nINPUT DATA=" << endl;
  layer1.p_data_cube->logical_print();

  cout << "\nINPUT MODEL=" << endl;
  layer1.p_model_cube->logical_print();

  forward.forward();

  LogicalCube<DataType_SFFloat, Layout_CRDB> out_expected(N-K+1, N-K+1, O, B);
  out_expected.reset_cube();
  simple_conv<DataType_SFFloat>(&data1, &kernel1, &out_expected);


  cout << "\nRESULT=" << endl;
  layer2.p_data_cube->logical_print();
  //layer2.p_data_cube->physical_print();

  cout << "\nEXPECTED RESULT=" << endl;
  out_expected.logical_print();
  forward.report_forward_last_transfer.print();
  forward.report_forward_history.print();
}

int main(int argc, const char * argv[]) {
  //TEST_LOWERING();

  TEST_BRIDGE();

  return 0;
}
