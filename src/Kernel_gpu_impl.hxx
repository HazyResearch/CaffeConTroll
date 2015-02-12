//
//  Kernel_impl_Lowering.hxx
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "common.h"
#include "cblas.h" // These two includes are from OpenBlas

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef moka_Kernel_GPU_impl_Lowering_hxx
#define moka_Kernel__GPUimpl_Lowering_hxx

#define SGEMM_ROWMAJOR(A,B,C,m,n,k,alpha,beta,transf_A,transf_B, lda, ldb, ldc) \
  BLASFUNC(sgemm)(transf_B, transf_A, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)

template <typename DataType, KernelConfig KERNELCONFIG>
Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_GEMM_GPU_CuBLAS, KERNELCONFIG>::
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
void Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_GEMM_GPU_CuBLAS, KERNELCONFIG>::
compute(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
    OutputLogicalCubeType * const p_output_cube) {

  // TODO
  // There is no doubt that this is slow...
  // This aims to be an end-to-end runnable version with correct anser
  // before we actually doing optimizations.

  Timer t;

  //std::cout << "Using new one!" << std::endl;
  cublasStatus_t status;
  cublasHandle_t handle;
  cublasCreate(&handle);

  t.restart();
  std::cout << "\nAllocating Device Memory..." << std::endl;
  float * M1, * M2, * M3;
  cudaMalloc((void**)&M1, p_input1_cube->n_elements * sizeof(DataType));
  cudaMalloc((void**)&M2, p_input2_cube->n_elements * sizeof(DataType));
  cudaMalloc((void**)&M3, p_output_cube->n_elements * sizeof(DataType));
  std::cout << "Allocation Done..." << t.elapsed() << std::endl;

  t.restart();
  std::cout << "\nCopy through PCIe..." << std::endl;
  cudaMemcpy(M1, p_input1_cube->p_data, p_input1_cube->n_elements*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(M2, p_input2_cube->p_data, p_input2_cube->n_elements*sizeof(DataType), cudaMemcpyHostToDevice);
  //cublasSetVector(p_input1_cube->n_elements, sizeof(DataType), p_input1_cube->p_data, 1, M1, 1);
  //cublasSetVector(p_input2_cube->n_elements, sizeof(DataType), p_input2_cube->p_data, 1, M2, 1);
  std::cout << "Copy Done..." << t.elapsed() << std::endl;
  std::cout << "Size = " << 1.0*(p_input1_cube->n_elements+p_input2_cube->n_elements)*sizeof(DataType)/1024/1024 << " MB" << std::endl;

  report_last_lowering.reset();

  int M, N, K;
  float _alpha = alpha;
  float _beta = beta;
  int LDA, LDB;
  cublasOperation_t N1, N2;

  if (KERNELCONFIG == KernelConfig_GEMM_NOTRANS_NOTRANS) {
    N1 = CUBLAS_OP_N;
    N2 = CUBLAS_OP_N;
    M = static_cast<int>(i1R);
    N = static_cast<int>(i2C);
    K = static_cast<int>(i1C);
    LDA = K;
    LDB = N;
  } else if (KERNELCONFIG == KernelConfig_GEMM_NOTRANS_TRANS) {
    N1 = CUBLAS_OP_N;
    N2 = CUBLAS_OP_T;
    M = static_cast<int>(i1R);
    N = static_cast<int>(i2R);
    K = static_cast<int>(i1C);
    LDA = K;
    LDB = K;
  } else if (KERNELCONFIG == KernelConfig_GEMM_TRANS_NOTRANS) {
    N1 = CUBLAS_OP_T;
    N2 = CUBLAS_OP_N;
    M = static_cast<int>(i1C);
    N = static_cast<int>(i2C);
    K = static_cast<int>(i1R);
    LDA = M;
    LDB = N;
  }

  t.restart();
  std::cout << "\nExecuting Kernel..." << std::endl;
  cudaStream_t stream1;
  cudaStreamCreate ( &stream1) ;
  cublasSetStream(handle, stream1);
  cublasSgemm(handle, N2, N1, N, M, K, &alpha, M2, LDB, M1, LDA, &beta, M3, N);
  cudaDeviceSynchronize();
  std::cout << "Kernel Exec Done..." << t.elapsed() << std::endl;

  t.restart();
  std::cout << "\nCopy result back..." << std::endl;
  cublasGetVector(p_output_cube->n_elements, sizeof(DataType), M3, 1, p_output_cube->p_data, 1);
  std::cout << "Copy done..." << t.elapsed() << std::endl;

  t.restart();
  std::cout << "\nFree GPU memory..." << std::endl;
  cudaFree(M1); cudaFree(M2); cudaFree(M3);
  std::cout << "Free done..." << t.elapsed() << std::endl;

  //BLASFUNC(sgemm)(&N2, &N1, &N, &M, &K, &_alpha, p_input2_cube->p_data,
  //    &LDB, p_input1_cube->p_data, &LDA, &_beta, p_output_cube->p_data, &N);

  report_last_lowering.end((i1R*i1C + i2R*i2C)*sizeof(DataType),
      oR*oC*sizeof(DataType), 1.0*M*N*K*2);
  report_history.aggregate(report_last_lowering);

}


#endif
