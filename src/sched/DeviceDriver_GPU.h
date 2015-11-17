
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "DeviceDriver.h"
#include <curand.h>

#ifndef _DEVICE_DRIVER_GPU_H
#define _DEVICE_DRIVER_GPU_H

class GPUDriver : public DeviceDriver{
public:

  int gpu_id = 0;

  // SHADJIS TODO: Rather than hard-code these use cudaGetDeviceProperties
  // E.g. for newer GPUs can use 1024 threads
  int max_cuda_blocks = 64000; // Actually 65535 is the max
  const int threadsPerBlock = 256;
  
  cublasStatus_t status;

  cublasHandle_t handle;

  // SHADJIS TODO: I don't think we need error (or status?) to be global
  // class members, they can probably be declared when they are needed.
  // This prevents any function modifying these from being declared const.
  cudaError_t err;

  GPUDriver();
  ~GPUDriver();

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
  void forward_bias(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const int fmap_size, const int depth, const int batch_size);
  void backward_bias(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const int fmap_size, const int depth, const int batch_size,
    const float *const device_ones);
  void backward_bias_fc(DeviceMemoryPointer * bias, DeviceMemoryPointer * output,
    const int D, const int B, const float *const device_ones);
  void maxpool_forward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _pool_forward_arg_helper args);
  void maxpool_backward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _pool_backward_arg_helper args);
  void lrn_forward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _lrn_forward_arg_helper args, const struct _lrn_forward_normalize_arg_helper args2);
  void lrn_backward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _lrn_backward_arg_helper args);
  void lower_cube_helper(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const struct PMapHelper args);
  void L1_update(const int n_elements, float * const p_gradient, const float lambda, const float * const p_model);

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

  void sgemm_new(const CBLAS_TRANSPOSE TA, const CBLAS_TRANSPOSE TB,
        const int M, const int N, const int K, const float alpha,
        const float * pA, const float * pB, const float beta, float * pC);
  void sgemv(const CBLAS_TRANSPOSE TA, const int M, const int N, const float alpha,
        const float * pA, const float * px, const float beta, float * py);

  template<FUNC_STRANSFORM func>
  void sapply(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry);

  void set_num_threads(const int nThreads);

  template<FUNC_SREDUCE func>
  void selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1,
    DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);

  void sinitialize_xavier(DeviceMemoryPointer *arr, const size_t n_batch);

  void sbernoulli_initialize(DeviceMemoryPointer *arr, const float p);
  
  void rand_uint_initialize(unsigned int * buf, const int n);

  void sgaussian_initialize(DeviceMemoryPointer *arr, const float mean, const float std_dev);

  void sconstant_initialize(DeviceMemoryPointer *arr, const float value);

  void * choose_ptr(void * host, void * device);

  using DeviceDriver::smath_apply_grad;
  
  void device_sync();
  void init_thread();
  void destroy_thread();
  void set_device_id(int id) { gpu_id = id; }
  // SHADJIS TODO: Some of these should be private
  void set_device() const; // Internal, sets device to current id and checks for error
  // SHADJIS TODO: currently I call set_device everywhere. Really we only need to call
  // it when the thread starts though.

private:

  random_device rd;
  curandGenerator_t curand_gen;
  curandStatus_t curand_err;
  
};

#endif

