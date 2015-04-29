
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

  template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
  void lower_cube(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const struct PMapHelper args);
    
  void inverse_lower_cube(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const struct _inverse_lower_cube_arg_helper args);

  template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
  void pmap2d_read_coalesce(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const struct PMapHelper args);

  template<FUNC_IDX_MAPPING f_dst_pos, FUNC_MM_MAPPING func>
  void parallel_map(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);

  void math_saxpy(const float alpha, DeviceMemoryPointer * X, DeviceMemoryPointer * Y) const;
  void math_saxpby(const float alpha, DeviceMemoryPointer * X, const float beta, DeviceMemoryPointer * Y) const;

  void math_saxpy(const int nElements, const float alpha, float * X, float * Y) const;
  void math_saxpby(const int nElements, const float alpha, float * X, const float beta, float * Y) const;

  void sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB,
        int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
        float beta, float * pC, int LDC);

  template<FUNC_STRANSFORM func>
  void sapply(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry);

  void set_num_threads(const int nThreads);

  template<FUNC_SREDUCE func>
  void selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1,
    DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);

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

