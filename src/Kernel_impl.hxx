//
//  Kernel_impl_Lowering.hxx
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "util.h" // include cblas routines

#ifndef moka_Kernel_impl_Lowering_hxx
#define moka_Kernel_impl_Lowering_hxx

template <typename DataType, KernelConfig KERNELCONFIG>
Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_GEMM_OpenBlas, KERNELCONFIG>::
Kernel(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
    const OutputLogicalCubeType * const p_output_cube, DeviceDriver * _p_driver) :
i1R(p_input1_cube->R), i1C(p_input1_cube->C), i1D(p_input1_cube->D), i1B(p_input1_cube->B),
i2R(p_input2_cube->R), i2C(p_input2_cube->C), i2D(p_input2_cube->D), i2B(p_input2_cube->B),
oR(p_output_cube->R), oC(p_output_cube->C), oD(p_output_cube->D), oB(p_output_cube->B),
alpha(1.0), beta(0), p_driver(_p_driver) {
  report_constructor.reset();
  report_last_lowering.reset();
  report_history.reset();
#ifdef _DO_ASSERT
  if (KERNELCONFIG == KernelConfig_GEMM_NOTRANS_NOTRANS) {
    assert(i1D==1); assert(i1B==1);
    assert(i2D==1); assert(i2B==1);
    assert(oD==1); assert(oB==1);
    assert(i1R==oR);
    assert(i1C==i2R);
    assert(i2C==oC);
  } else if (KERNELCONFIG == KernelConfig_GEMM_NOTRANS_TRANS) {
    assert(i1D==1); assert(i1B==1);
    assert(i2D==1); assert(i2B==1);
    assert(oD==1); assert(oB==1);
    assert(i1R==oR);
    assert(i1C==i2C);
    assert(i2R==oC);
  } else if (KERNELCONFIG == KernelConfig_GEMM_TRANS_NOTRANS) {
    assert(i1D==1); assert(i1B==1);
    assert(i2D==1); assert(i2B==1);
    assert(oD==1); assert(oB==1);
    assert(i1C==oR);
    assert(i1R==i2R);
    assert(i2C==oC);
  } else {
    std::cerr << "ERROR: Unsupported KernelConfig for GEMM." << std::endl;
    assert(false);
  }
#endif
  report_constructor.end(0, 0, 0);
}

template <typename DataType, KernelConfig KERNELCONFIG>
void Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_GEMM_OpenBlas, KERNELCONFIG>::
compute(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
    OutputLogicalCubeType * const p_output_cube) {

  report_last_lowering.reset();

  int M, N, K;
  float _alpha = alpha;
  float _beta = beta;
  int LDA, LDB;
  CBLAS_TRANSPOSE TA,TB;

  // The convention is that
  // input_cube_1 is A and
  // input_cube_2 is B
  // This code is doing alpha* A*B + beta*C
  // cout << "HERE: " << KERNELCONFIG << std::endl;
  // cout << "\tA : " << i1R << " x " << i1C << std::endl;
  // cout << "\tB : " << i2R << " x " << i2C << std::endl;
  // cout << "\tC : " << oR << " x " << oC << std::endl;

  if (KERNELCONFIG == KernelConfig_GEMM_NOTRANS_NOTRANS) {
    TA = CblasNoTrans;
    TB = CblasNoTrans;
    M = i1R; // rows in A and C (sgemm spec)
    N = i2C; // cols in B and C
    K = i1C; // cols in A or rows in B
    LDA = i1C; //
    LDB = i2C; //
  } else if (KERNELCONFIG == KernelConfig_GEMM_NOTRANS_TRANS) {
    TA = CblasNoTrans;
    TB = CblasTrans;
    M = i1R; // A is not transposed
    N = i2R; // B is transposed
    K = i1C; // cols in A not transposed
    LDA = i1C; // columns in A
    LDB = i2C; // B is transposed
  } else if (KERNELCONFIG == KernelConfig_GEMM_TRANS_NOTRANS) {
    TA = CblasTrans;
    TB = CblasNoTrans;
    M = i1C; // A is transposed
    N = i2C; // B is not transposed
    K = i1R; // A is transposed
    LDA = i1C; // A is transposed
    LDB = i2C; // not transposed
  }

  // we reverse A and B here by default
  p_driver->sgemm(CblasRowMajor, TA, TB, M, N, K, _alpha,
        p_input1_cube->get_p_data(), LDA,
        p_input2_cube->get_p_data(), LDB,
        _beta, p_output_cube->get_p_data(), N);

  report_last_lowering.end((i1R*i1C + i2R*i2C)*sizeof(DataType),
      oR*oC*sizeof(DataType), 1.0*M*N*K*2);
  report_history.aggregate(report_last_lowering);

}

template <typename DataType, KernelConfig KERNELCONFIG>
Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KERNELCONFIG>::
Kernel(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
    const OutputLogicalCubeType * const p_output_cube, DeviceDriver * _p_driver) :
  i1n_elements(p_input1_cube->n_elements),
  i2n_elements(p_input2_cube->n_elements),
  on_elements(p_output_cube->n_elements),
  p_driver(_p_driver) {
  report_constructor.reset();
  report_last_lowering.reset();
  report_history.reset();
#ifdef _DO_ASSERT
  assert(i1n_elements==i2n_elements);
  assert(i1n_elements==on_elements);
#endif
  report_constructor.end(0, 0, 0);
}

template <typename DataType, KernelConfig KERNELCONFIG>
void Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KERNELCONFIG>::
compute(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
    OutputLogicalCubeType * const p_output_cube) {

  report_last_lowering.reset();

  DeviceMemoryPointer * input1 = p_input1_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * input2 = p_input1_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = p_output_cube->get_device_pointer(p_driver);

  if (KERNELCONFIG == KernelConfig_NONE){
    const auto func = [](DataType a, DataType b)->DataType{return a*b;};
    p_driver->selementwise_reduce2(*output, *input1, *input2, func);
  }else if(KernelConfig_TANHGRAD_ON_INPUT1){
    const auto func = [](DataType a, DataType b)->DataType{return (1-a*a)*b;};
    p_driver->selementwise_reduce2(*output, *input1, *input2, func);
  }else{
      std::cerr << "ERROR: Not supported KernelConfig!" << std::endl;
      assert(false);
  }

  // Following is the old physical code.
  /*
  size_t i = 0; // TODO: change to SIMD (actuall the following one is so easy to be vectorized by the compiler with -O3...)
  DataType * const output_data = p_output_cube->get_p_data();
  const DataType * const input1_data = p_input1_cube->get_p_data();
  const DataType * const input2_data = p_input1_cube->get_p_data();
  for (i = 0; i < i1n_elements; i++) {
    if (KERNELCONFIG == KernelConfig_NONE) { // this should be optimized out by the compiler with -O3
      output_data[i] = input1_data[i] * input2_data[i];
    } else if (KERNELCONFIG == KernelConfig_TANHGRAD_ON_INPUT1) {
      output_data[i] =
        (1 - input1_data[i] * input1_data[i]) * input2_data[i];
    } else {
      std::cerr << "ERROR: Not supported KernelConfig!" << std::endl;
      assert(false);
    }
  }
  */

  double flop = 1.0*i1n_elements;
  if (KERNELCONFIG == KernelConfig_TANHGRAD_ON_INPUT1) {
    flop = 1.0*i1n_elements*3;
  }

  report_last_lowering.end(i1n_elements*2*sizeof(DataType),
      i1n_elements*sizeof(DataType), flop);
  report_history.aggregate(report_last_lowering);
}

#endif





