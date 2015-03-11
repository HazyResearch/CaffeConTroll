
#include "DeviceDriver.h"
#include "cblas.h"
#include "../kernels/lowering.h"
#include "../kernels/lowering.hxx"

#ifndef _DEVICE_DRIVER_CPU_H
#define _DEVICE_DRIVER_CPU_H

class CPUDriver : public DeviceDriver{
public:

  CPUDriver();

  DeviceMemoryPointer * get_device_pointer(void * ptr, size_t size_in_byte);

  void malloc(DeviceMemoryPointer * dst);

  void free(DeviceMemoryPointer * dst);

  void memcpy(DeviceMemoryPointer * dst, DeviceMemoryPointer * src);

  void memset(DeviceMemoryPointer * dst, const char value);

  template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
  void pmap2d_read_coalesce(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct PMapHelper args);

  template<FUNC_IDX_MAPPING f_dst_pos, FUNC_MM_MAPPING func>
  void parallel_map(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);

  void smath_axpy(const float alpha, DeviceMemoryPointer * X, DeviceMemoryPointer * Y);

  void smath_axpby(const float alpha, DeviceMemoryPointer * X, const float beta, DeviceMemoryPointer * Y) ;

  void sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
        int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
        float beta, float * pC, int LDC);

  template<FUNC_STRANSFORM func>
  void sapply(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry);

  void set_num_threads(const int nThreads);

  template<FUNC_SREDUCE func>
  void selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1, 
    DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);

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




