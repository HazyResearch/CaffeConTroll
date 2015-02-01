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
#include "bridges/DropoutBridge.h"
#include "Layer.h"
//#include "bridges/ParallelizedBridge.h"
#include "parser/parser.h"
#include "parser/corpus.h"
#include "util.h"

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

  Util::xavier_initialize(data1.p_data, N*N*D*B, B);
  Util::constant_initialize<float>(bias.p_data, 0.0, O);

  kernel1.reset_cube();
  for (size_t i = 0; i < K*K*D*O; i++) {
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
  for(int i=0; i < iD*mB; i++) {
    data1.p_data[i] = (rand()%5)*0.1;
  }
  cout << "INPUT DATA:" << endl;
  data1.logical_print();

  srand(0);
  for(int n=0;n<mB;n++) {
    label.p_data[n] = rand()%10;
  }
  cout << "LABEL DATA:" << endl;
  label.logical_print();

  softmaxBridge_.forward();
  cout << "LOSS: " << softmaxBridge_.loss << endl;
}

void LeNet(const char * file) {
  cnn::SolverParameter solver_param;
  Parser::ReadProtoFromTextFile(file, &solver_param);
  cnn::NetParameter net_param;
  Parser::ReadNetParamsFromTextFile(solver_param.net(), &net_param);
  // Build Network
  cnn::Datum train_data;

  //cnn::Datum test_data;
  //int n_label = 10;

  //int mini_batch_size_test;
  //size_t noutput_feature_map, nrow_output, ncol_output;
  //int nrow_conv, ncol_conv;
  //int pad, stride;

  //int local_size;
  //float ratio, alpha, beta;
  //bool is_across = false;

  const size_t num_layers = net_param.layers_size();
  const Corpus* corpus = NULL;

  // load training data into corpus
  for (size_t i = 0; i < num_layers; ++i) {
    cnn::LayerParameter layer_param = net_param.layers(i);
    if (layer_param.type() == cnn::LayerParameter_LayerType_DATA) {
      if (layer_param.include(0).phase() == 0) { // training phase
        Parser::DataSetup(layer_param, train_data);
        corpus = new Corpus(layer_param);
        cout << "Corpus train loaded" << endl;
        break;
      }
      //if (layer_param.include(0).phase() == 1) { // testing phase
      //  dataSetup(layer_param, test_data);
      //  mini_batch_size_test = layer_param.data_param().batch_size();
      //  mini_batch_size_test = mini_batch_size_train;
      //  //Corpus corpus(layer_param);
      //  //cout << "Corpus test loaded" << endl;
      //}
      //num_layers--;
    }
  }

#ifdef _DO_ASSERT
  assert(corpus != NULL);
#endif

  cout << "NUM IMAGES: " << corpus->n_images << endl;
  cout << "NUM ROWS: " << corpus->n_rows << endl;
  cout << "NUM COLS: " << corpus->n_cols << endl;
  cout << "NUM CHANNELS: " << corpus->dim << endl;
  cout << "MINI BATCH SIZE: " << corpus->mini_batch_size << endl;
  cout << "NUM MINI BATCHES: " << corpus->num_mini_batches << endl;
  cout << "LAST BATCH SIZE: " << corpus->last_batch_size << endl;

  Timer t = Timer();
  const size_t num_epochs = 6;
  const size_t R1 = corpus->n_rows;
  const size_t C1 = corpus->n_cols;
  const size_t D = corpus->dim;
  const size_t B = corpus->mini_batch_size;
  const size_t last_B = corpus->last_batch_size;
  const size_t conv_O1 = 20;
  const size_t conv_O2 = 50;
  const size_t conv_O3 = 500;
  const size_t conv_O4 = 10;
  const size_t conv_K = 5;
  const size_t pool_K = 2;
  const size_t pool_stride = 2;

  // Layers:
  // conv1 (O: 20, K: 5),
  // pool1 (kernel: 2, stride: 2),
  // conv2 (O: 50, K: 5),
  // pool2 (kernel: 2, stride: 2),
  // ip1 (O: 500, K:1),
  // relu1,
  // ip2 (O: 10, K: 1),
  // softmax
  // LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(conv_K, conv_K, D, conv_O1);
  // LogicalCube<DataType_SFFloat, Layout_CRDB> bias1(1, 1, conv_O1, 1);
  // Util::xavier_initialize(kernel1.p_data, conv_K*conv_K*D*conv_O1, conv_O1);
  // Util::constant_initialize<float>(bias1.p_data, 0.0, conv_O1);

  // LogicalCube<DataType_SFFloat, Layout_CRDB> kernel2(conv_K, conv_K, conv_O1, conv_O2);
  // LogicalCube<DataType_SFFloat, Layout_CRDB> bias2(1, 1, conv_O2, 1);
  // Util::xavier_initialize(kernel2.p_data, conv_K*conv_K*conv_O1*conv_O2, conv_O2);
  // Util::constant_initialize<float>(bias2.p_data, 0.0, conv_O2);

  // LogicalCube<DataType_SFFloat, Layout_CRDB> kernel3(1, 1, conv_O2, conv_O3);
  // LogicalCube<DataType_SFFloat, Layout_CRDB> bias3(1, 1, conv_O3, 1);
  // Util::xavier_initialize(kernel3.p_data, 1*1*conv_O2*conv_O3, conv_O3);
  // Util::constant_initialize<float>(bias3.p_data, 0.0, conv_O3);

  // LogicalCube<DataType_SFFloat, Layout_CRDB> kernel4(1, 1, conv_O3, conv_O4);
  // LogicalCube<DataType_SFFloat, Layout_CRDB> bias4(1, 1, conv_O4, 1);
  // Util::xavier_initialize(kernel4.p_data, 1*1*conv_O3*conv_O4, conv_O4);
  // Util::constant_initialize<float>(bias4.p_data, 0.0, conv_O4);

  LogicalCube<DataType_SFFloat, Layout_CRDB> data1(NULL, R1, C1, D, B); // must be initialized to point to next mini batch
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad1(R1, C1, D, B);
  // conv1
  size_t R2 = R1-conv_K+1, C2 = C1-conv_K+1;
  LogicalCube<DataType_SFFloat, Layout_CRDB> data2(R2, C2, conv_O1, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad2(R2, C2, conv_O1, B);
  // pool1
  size_t R3 = (R2-pool_K)/pool_stride + 1, C3 = (C2-pool_K)/pool_stride + 1;
  LogicalCube<DataType_SFFloat, Layout_CRDB> data3(R3, C3, conv_O1, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad3(R3, C3, conv_O1, B);
  // conv2
  size_t R4 = R3-conv_K+1, C4 = C3-conv_K+1;
  LogicalCube<DataType_SFFloat, Layout_CRDB> data4(R4, C4, conv_O2, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad4(R4, C4, conv_O2, B);
  // pool2
  size_t R5 = (R4-pool_K)/pool_stride + 1, C5 = (C4-pool_K)/pool_stride + 1;
  LogicalCube<DataType_SFFloat, Layout_CRDB> data5(R5, C5, conv_O2, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad5(R5, C5, conv_O2, B);
  // ip1
  LogicalCube<DataType_SFFloat, Layout_CRDB> data6(1, 1, conv_O3, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad6(1, 1, conv_O3, B);
  // relu1
  LogicalCube<DataType_SFFloat, Layout_CRDB> data7(1, 1, conv_O3, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad7(1, 1, conv_O3, B);
  // ip2
  LogicalCube<DataType_SFFloat, Layout_CRDB> data8(1, 1, conv_O4, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad8(1, 1, conv_O4, B);
  // softmax
  LogicalCube<DataType_SFFloat, Layout_CRDB> data9(NULL, 1, 1, conv_O4, B); // not used
  LogicalCube<DataType_SFFloat, Layout_CRDB> grad9(1, 1, conv_O4, B);
  LogicalCube<DataType_SFFloat, Layout_CRDB> labels(NULL, 1, 1, 1, B); // must be initialized to point to next mini batch

  LogicalCube<DataType_SFFloat, Layout_CRDB> kernel1(conv_K, conv_K, D, conv_O1);
  LogicalCube<DataType_SFFloat, Layout_CRDB> bias1(1, 1, conv_O1, 1);
  Util::xavier_initialize(kernel1.p_data, conv_K*conv_K*D*conv_O1, conv_O1);
  Util::constant_initialize<float>(bias1.p_data, 0.0, conv_O1);

  LogicalCube<DataType_SFFloat, Layout_CRDB> kernel2(conv_K, conv_K, conv_O1, conv_O2);
  LogicalCube<DataType_SFFloat, Layout_CRDB> bias2(1, 1, conv_O2, 1);
  Util::xavier_initialize(kernel2.p_data, conv_K*conv_K*conv_O1*conv_O2, conv_O2);
  Util::constant_initialize<float>(bias2.p_data, 0.0, conv_O2);

  LogicalCube<DataType_SFFloat, Layout_CRDB> kernel3(R5, C5, conv_O2, conv_O3);
  LogicalCube<DataType_SFFloat, Layout_CRDB> bias3(1, 1, conv_O3, 1);
  Util::xavier_initialize(kernel3.p_data, R5*C5*conv_O2*conv_O3, conv_O3);
  Util::constant_initialize<float>(bias3.p_data, 0.0, conv_O3);

  LogicalCube<DataType_SFFloat, Layout_CRDB> kernel4(1, 1, conv_O3, conv_O4);
  LogicalCube<DataType_SFFloat, Layout_CRDB> bias4(1, 1, conv_O4, 1);
  Util::xavier_initialize(kernel4.p_data, 1*1*conv_O3*conv_O4, conv_O4);
  Util::constant_initialize<float>(bias4.p_data, 0.0, conv_O4);

  Layer<DataType_SFFloat, Layout_CRDB> layer1(&data1, &grad1);
  Layer<DataType_SFFloat, Layout_CRDB> layer2(&data2, &grad2);
  Layer<DataType_SFFloat, Layout_CRDB> layer3(&data3, &grad3);
  Layer<DataType_SFFloat, Layout_CRDB> layer4(&data4, &grad4);
  Layer<DataType_SFFloat, Layout_CRDB> layer5(&data5, &grad5);
  Layer<DataType_SFFloat, Layout_CRDB> layer6(&data6, &grad6);
  Layer<DataType_SFFloat, Layout_CRDB> layer7(&data7, &grad7);
  Layer<DataType_SFFloat, Layout_CRDB> layer8(&data8, &grad8);
  Layer<DataType_SFFloat, Layout_CRDB> layer9(&data9, &grad9);

  ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>
    conv1(&layer1, &layer2, &kernel1, &bias1);
  MaxPoolingBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>
    pool1(&layer2, &layer3, new BridgeConfig(pool_K, pool_stride));
  ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>
    conv2(&layer3, &layer4, &kernel2, &bias2);
  MaxPoolingBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>
    pool2(&layer4, &layer5, new BridgeConfig(pool_K, pool_stride));
  ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>
    ip1(&layer5, &layer6, &kernel3, &bias3);
  ReLUBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>
    relu1(&layer6, &layer7);
  ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>
    ip2(&layer7, &layer8, &kernel4, &bias4);
  SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>
    softmax(&layer8, &layer9, &labels);

  for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
    cout << "EPOCH: " << epoch << endl;
    float epoch_loss = 0.0;
    // num_mini_batches - 1, because we need one more iteration for the final mini batch
    // (the last mini batch may not be the same size as the rest of the mini batches)
    size_t corpus_batch_index = 0;
    for (size_t batch = 0; batch < corpus->num_mini_batches - 1; ++batch) {
      if( batch % 100 == 0){
        cout << "BATCH: " << batch << endl;  
      }
      
      // initialize data1 for this mini batch
      float * const mini_batch = corpus->images->physical_get_RCDslice(corpus_batch_index);
      data1.p_data = mini_batch;
      //data1.logical_print();
      // reset loss
      softmax.loss = 0.0;

      // initialize labels for this mini batch
      labels.p_data = corpus->labels->physical_get_RCDslice(corpus_batch_index);
      corpus_batch_index += B;
      // clear data and grad outputs for this batch (but not the weights and biases!)
      grad1.reset_cube(); data2.reset_cube(); grad2.reset_cube(); data3.reset_cube(); grad3.reset_cube();
      data4.reset_cube(); grad4.reset_cube(); data5.reset_cube(); grad5.reset_cube(); data6.reset_cube();
      grad6.reset_cube(); data7.reset_cube(); grad7.reset_cube(); data8.reset_cube(); grad8.reset_cube();
      Util::constant_initialize<float>(grad9.p_data, 1.0, 1*1*conv_O4*B); //initialize to 1 for backprop

      //cout << "FORWARD PASS" << endl;
      // forward pass
      conv1.forward();
      //cout << "conv1" << endl;
      pool1.forward();
      //cout << "pool1" << endl;
      conv2.forward();
      //cout << "conv2" << endl;
      pool2.forward();
      //cout << "pool2" << endl;
      ip1.forward();
      //cout << "ip1" << endl;
      relu1.forward();
      //cout << "relu1" << endl;
      ip2.forward();
      //cout << "ip2" << endl;
      softmax.forward();
      //cout << "softmax" << endl;
      epoch_loss += (softmax.loss/B);
      //cout << "LOSS: " << (softmax.loss/B) << endl;
      //cout << "BACKWARD PASS" << endl;
      // backward pass
      softmax.backward();
      //cout << "softmax" << endl;
      ip2.backward();
      //cout << "ip2" << endl;
      relu1.backward();
      //cout << "relu" << endl;
      ip1.backward();
      //cout << "ip1" << endl;
      pool2.backward();
      //cout << "pool2" << endl;
      conv2.backward();
      //cout << "conv2" << endl;
      pool1.backward();
      //cout << "pool1" << endl;
      conv1.backward();
      
      //cout << "conv1" << endl;
    }
    cout << "LOSS:" << epoch_loss/(num_mini_batches-1) << endl;
    // compute very last batch
    // data1.B = last_B; grad1.B = last_B;
    // data2.B = last_B; grad2.B = last_B;
    // data3.B = last_B; grad3.B = last_B;
    // data4.B = last_B; grad4.B = last_B;
    // data5.B = last_B; grad5.B = last_B;
    // data6.B = last_B; grad6.B = last_B;
    // data7.B = last_B; grad7.B = last_B;
    // data8.B = last_B; grad8.B = last_B;
    // data9.B = last_B; grad9.B = last_B;
    // float * const mini_batch = corpus->images->physical_get_RCDslice(corpus_batch_index);
    // data1.p_data = mini_batch;

    // // reset loss
    // softmax.loss = 0.0;

    // // initialize labels for this mini batch
    // labels.p_data = corpus->labels->physical_get_RCDslice(corpus_batch_index);

    // // clear data and grad outputs for this batch (but not the weights and biases!)
    // grad1.reset_cube(); data2.reset_cube(); grad2.reset_cube(); data3.reset_cube(); grad3.reset_cube();
    // data4.reset_cube(); grad4.reset_cube(); data5.reset_cube(); grad5.reset_cube(); data6.reset_cube();
    // grad6.reset_cube(); data7.reset_cube(); grad7.reset_cube(); data8.reset_cube(); grad8.reset_cube();
    // Util::constant_initialize<float>(grad9.p_data, 1.0, R5*C5*conv_O4*B); //initialize to 1 for backprop

    // //cout << "FORWARD PASS" << endl;
    // // forward pass
    // conv1.forward();
    // //cout << "conv1" << endl;
    // pool1.forward();
    // //cout << "pool1" << endl;
    // conv2.forward();
    // //cout << "conv2" << endl;
    // pool2.forward();
    // //cout << "pool2" << endl;
    // ip1.forward();
    // //cout << "ip1" << endl;
    // relu1.forward();
    // //cout << "relu1" << endl;
    // ip2.forward();
    // //cout << "ip2" << endl;
    // softmax.forward();
    // //cout << "softmax" << endl;

    // //cout << "LOSS: " << softmax.loss << endl;

    // //cout << "BACKWARD PASS" << endl;
    // // backward pass
    // softmax.backward();
    // //cout << "softmax" << endl;
    // ip2.backward();
    // //cout << "ip2" << endl;
    // relu1.backward();
    // //cout << "relu" << endl;
    // ip1.backward();
    // //cout << "ip1" << endl;
    // pool2.backward();
    // //cout << "pool2" << endl;
    // conv2.backward();
    // //cout << "conv2" << endl;
    // pool1.backward();
    // //cout << "pool1" << endl;
    // conv1.backward();
    //cout << "conv1" << endl;
    cout << "Time Elapsed for a single epoch: " << t.elapsed() << endl;
  }
  cout << "Total Time Elapsed: " << t.elapsed() << endl;
}

int main(int argc, const char * argv[]) {
  //TEST_LOWERING();

  //TEST_CONVOLUTION_BRIDGE();

  //TEST_SOFTMAX();

  if (argc != 2) {
    cout << "Usage: ./deepnet <solver.prototxt>" << endl;
    exit(1);
  }

  LeNet(argv[1]);

  return 0;
}
