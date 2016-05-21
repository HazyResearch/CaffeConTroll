
#include "DeviceDriver.h"
#include "cblas.h"

#ifndef _DEVICE_DRIVER_CPU_H
#define _DEVICE_DRIVER_CPU_H

class CPUDriver : public DeviceDriver {
public:

  CPUDriver();
  ~CPUDriver();

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
  void backward_bias(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const int fmap_size, const int depth, const int batch_size, 
    const float *const device_ones);
  void backward_bias_fc(DeviceMemoryPointer * bias, DeviceMemoryPointer * output,
    const int D, const int B, const float *const device_ones);
  void maxpool_forward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _pool_forward_arg_helper args);
  void maxpool_backward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _pool_backward_arg_helper args);
  void avepool_forward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _pool_forward_arg_helper args);
  void avepool_backward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _pool_backward_arg_helper args);
  void lrn_forward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _lrn_forward_arg_helper args, const struct _lrn_forward_normalize_arg_helper args2);
  void lrn_backward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _lrn_backward_arg_helper args);
  void L1_update(const int n_elements, float * const p_gradient, 
    const float lambda, const float * const p_model);
    
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
  void math_saxpby(const int nElements, const float alpha, const float * X, const float beta, float * Y) const;


  void sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB,
        int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
        float beta, float * pC, int LDC);

  void sgemm_new(const CBLAS_TRANSPOSE TA, const CBLAS_TRANSPOSE TB,
        const int M, const int N, const int K, const float alpha,
        const float * pA, const float * pB, const float beta, float * pC);
  void sgemv(const CBLAS_TRANSPOSE TA, const int M, const int N, const float alpha,
        const float * pA, const float * px, const float beta, float * py);

  void sscale(const int n, const float alpha, const float *x, float* y);
  void sscale_inplace(const int n, const float alpha, float *x);
  void eltwise_mul(const int n, const float* const a, const float* const b, float* y);
  void eltwise_powx(const int n, const float* a, const float b, float* y);
  void eltwise_pow2(const int n, const float* a, float* y);
  void eltwise_sqrt(const int n, const float* a, float* y);
  void add_scalar(const int N, const float alpha, float* Y);
  void eltwise_div(const int n, const float* a, const float* b, float* y);
  float dot_prod(const int n, const float* x, const float* y);

  template<FUNC_STRANSFORM func>
  void sapply(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry);

  void set_num_threads(const int nThreads);

  template<FUNC_SREDUCE func>
  void selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1,
    DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);

  FUNC_STRANSFORM * srand_uni(float lower, float upper, DeviceMemoryPointer * arg);

  FUNC_STRANSFORM * srand_bern(float p, DeviceMemoryPointer * arg);

  FUNC_STRANSFORM * srand_gaussian(float mean, float std_dev, DeviceMemoryPointer * arg);

  void init_rng(const int random_seed);
  
  void sinitialize_xavier(DeviceMemoryPointer *arr, const size_t n_batch, const int random_seed = -1);

  void sbernoulli_initialize(DeviceMemoryPointer *arr, const float p, const int random_seed = -1);

  void sgaussian_initialize(DeviceMemoryPointer *arr, const float mean, const float std_dev, const int random_seed = -1);

  void sconstant_initialize(DeviceMemoryPointer *arr, const float value);

  void * choose_ptr(void * host, void * device);

  using DeviceDriver::smath_apply_grad;

private:

  random_device rd;
};

#endif




