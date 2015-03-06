
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

#include "DeviceDriver.h"

#ifndef _DEVICE_DRIVER_GPU_H
#define _DEVICE_DRIVER_GPU_H

class GPUDriver : public DeviceDriver{
public:

  int gpu_id = 0;

  int threadsPerBlock = 256;

  cublasStatus_t status;
  
  cublasHandle_t handle;

  cudaError_t err;

  GPUDriver();

  DeviceMemoryPointer * get_device_pointer(void * ptr, size_t size_in_byte);

  void malloc(DeviceMemoryPointer * dst);

  void free(DeviceMemoryPointer * dst);

  void memcpy(DeviceMemoryPointer * dst, DeviceMemoryPointer * src);

  void memset(DeviceMemoryPointer * dst, const char value);

  void parallel_map(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    size_t src_skip, FUNC_IDX_MAPPING * f_dst_pos, DeviceMemoryPointer * const f_dst_pos_curry,
    FUNC_MM_MAPPING * func, DeviceMemoryPointer * const func_curry);

  void smath_axpy(const float alpha, DeviceMemoryPointer * X, DeviceMemoryPointer * Y);

  void smath_axpby(const float alpha, DeviceMemoryPointer * X, const float beta, DeviceMemoryPointer * Y) ;

  void sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
        int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
        float beta, float * pC, int LDC);

  void sapply(DeviceMemoryPointer * dst, FUNC_STRANSFORM * func, DeviceMemoryPointer * const func_curry);

  void set_num_threads(const int nThreads);

  void selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1, 
    DeviceMemoryPointer * src2, FUNC_SREDUCE * func, DeviceMemoryPointer * const func_curry);

  FUNC_STRANSFORM * srand_uni(float lower, float upper, DeviceMemoryPointer * arg);

  FUNC_STRANSFORM * srand_bern(float p, DeviceMemoryPointer * arg);

  FUNC_STRANSFORM * srand_gaussian(float mean, float std_dev, DeviceMemoryPointer * arg);

  void sinitialize_xavier(DeviceMemoryPointer *arr, const size_t n_batch);

  void sbernoulli_initialize(DeviceMemoryPointer *arr, const float p);

  void sgaussian_initialize(DeviceMemoryPointer *arr, const float mean, const float std_dev);
  
  void sconstant_initialize(DeviceMemoryPointer *arr, const float value);

  void * choose_ptr(void * host, void * device);

  using DeviceDriver::smath_apply_grad;

private:

  random_device rd;
};

#endif

