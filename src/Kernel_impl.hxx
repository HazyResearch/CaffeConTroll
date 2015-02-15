//
//  Kernel_impl_Lowering.hxx
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "common.h"
#include "cblas.h" // These two includes are from OpenBlas

#ifndef moka_Kernel_impl_Lowering_hxx
#define moka_Kernel_impl_Lowering_hxx

#define SGEMM_ROWMAJOR(A,B,C,m,n,k,alpha,beta,transf_A,transf_B, lda, ldb, ldc) \
  BLASFUNC(sgemm)(transf_B, transf_A, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)

template <typename DataType, KernelConfig KERNELCONFIG>
Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_GEMM_OpenBlas, KERNELCONFIG>::
Kernel(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
    const OutputLogicalCubeType * const p_output_cube) :
i1R(p_input1_cube->R), i1C(p_input1_cube->C), i1D(p_input1_cube->D), i1B(p_input1_cube->B),
i2R(p_input2_cube->R), i2C(p_input2_cube->C), i2D(p_input2_cube->D), i2B(p_input2_cube->B),
oR(p_output_cube->R), oC(p_output_cube->C), oD(p_output_cube->D), oB(p_output_cube->B),
alpha(1.0), beta(0) {
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
    std::cout << i1R << "   " << i1C << std::endl;
    std::cout << i2R << "   " << i2C << std::endl;
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
  char N1, N2;

  if (KERNELCONFIG == KernelConfig_GEMM_NOTRANS_NOTRANS) {
    N1 = 'N';
    N2 = 'N';
    M = static_cast<int>(i1R);
    N = static_cast<int>(i2C);
    K = static_cast<int>(i1C);
    LDA = K;
    LDB = N;
  } else if (KERNELCONFIG == KernelConfig_GEMM_NOTRANS_TRANS) {
    N1 = 'N';
    N2 = 'T';
    M = static_cast<int>(i1R);
    N = static_cast<int>(i2R);
    K = static_cast<int>(i1C);
    LDA = K;
    LDB = K;
  } else if (KERNELCONFIG == KernelConfig_GEMM_TRANS_NOTRANS) {
    N1 = 'T';
    N2 = 'N';
    M = static_cast<int>(i1C);
    N = static_cast<int>(i2C);
    K = static_cast<int>(i1R);
    LDA = M;
    LDB = N;  
  }

  BLASFUNC(sgemm)(&N2, &N1, &N, &M, &K, &_alpha, p_input2_cube->p_data,
      &LDB, p_input1_cube->p_data, &LDA, &_beta, p_output_cube->p_data, &N);

  report_last_lowering.end((i1R*i1C + i2R*i2C)*sizeof(DataType),
      oR*oC*sizeof(DataType), 1.0*M*N*K*2);
  report_history.aggregate(report_last_lowering);

}

template <typename DataType, KernelConfig KERNELCONFIG>
Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KERNELCONFIG>::
Kernel(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
    const OutputLogicalCubeType * const p_output_cube) :
  i1n_elements(p_input1_cube->n_elements),
  i2n_elements(p_input2_cube->n_elements),
  on_elements(p_output_cube->n_elements) {
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

  size_t i = 0; // TODO: change to SIMD (actuall the following one is so easy to be vectorized by the compiler with -O3...)
  for (i=0;i<i1n_elements;i++) {
    if (KERNELCONFIG == KernelConfig_NONE) { // this should be optimized out by the compiler with -O3
      p_output_cube->p_data[i] =
        p_input1_cube->p_data[i] * p_input2_cube->p_data[i];
    } else if (KERNELCONFIG == KernelConfig_TANHGRAD_ON_INPUT1) {
      p_output_cube->p_data[i] =
        (1-p_input1_cube->p_data[i]*p_input1_cube->p_data[i]) * p_input2_cube->p_data[i];
    } else {
      std::cerr << "ERROR: Not supported KernelConfig!" << std::endl;
      assert(false);
    }
  }

  double flop = 1.0*i1n_elements;
  if (KERNELCONFIG == KernelConfig_TANHGRAD_ON_INPUT1) {
    flop = 1.0*i1n_elements*3;
  }

  report_last_lowering.end(i1n_elements*2*sizeof(DataType),
      i1n_elements*sizeof(DataType), flop);
  report_history.aggregate(report_last_lowering);
}

#endif
