
#include <functional>
#include <math.h>
#include <random>
#include "DeviceMemoryPointer.h"
#include "cblas.h"

#ifndef _DEVICE_DRIVER_H
#define _DEVICE_DRIVER_H

using std::max;
using std::min;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;
using std::bernoulli_distribution;
using std::normal_distribution;

/**
 * We use C style function pointer here just becauses
 * there is no clean way in CUDA (7.0) and OpenCL (2.0)
 * to pass a host C++11 Lambda with capture to the device.
 * We could have used C-style macro to achieve this,
 * but that is even more messier.
 **/
typedef size_t (*FUNC_IDX_MAPPING) (size_t, void * const);
typedef void (*FUNC_MM_MAPPING) (void *, void *, void * const);
typedef float (*FUNC_STRANSFORM) (float, void * const);
typedef float (*FUNC_SREDUCE) (float, float, void * const);


__host__ __device__ float __sconstant_initialize_helper(float a, void * arg){
  return *((float*)arg);
}
__device__ FUNC_STRANSFORM _sconstant_initialize_helper = __sconstant_initialize_helper;


/**
 * A DeviceDriver is the only way
 * that CcT talks to a certain device
 * to invoke computation and data
 * movement *INSIDE* a device. 
 *
 * Given a DeviceDriver, all Bridges
 * should be purely *logical*. This is
 * the goal of introducing DeviceDriver.
 *
 * All cross device operation needs to
 * be done via derefercing a DeviceMemoryPointer,
 * it is not DeviceDriver's job to worry
 * about cross-device data movement.
 *
 * A DeviceDriver needs to provide certain
 * interface, e.g., how BLAS can be called,
 * or different helper functions, e.g.,
 * axpy.
 * 
 * One question is what function should we
 * put in DeviceDriver and what function should
 * we put in Util? The answer is that
 * Util contains all functions that are
 * device-independent, e.g., get_learning_rate,
 * and DeviceDriver contains all functions
 * that are device-dependent. 
 *
 * Error handelling. All functions return void, however
 * if error occurs, it assert(false). In short, we
 * assume it is the worker's repsonbility to deal
 * with error, instead of the caller.
 *
 * TODO:
 *  - Template this by double, float etc.
 **/
class DeviceDriver{
public:

  /**
   * A UDF that can be called by a driver might see on
   * host or might sit on device. It is driver's responsiblity
   * to choose one.
   *
   * Note that, a single function should have only one 
   * implementation. It is on device not by copy&paste
   * the code, instead, by a very thin wrapper, e.g.,
   *   __device__ T func_on_device = func_on_host.
   **/
  virtual void * choose_ptr(void * host, void * device) = 0;

  virtual DeviceMemoryPointer * get_device_pointer(void * ptr, size_t size_in_byte) = 0;

  /**
   * Memory manipulation and data movement.
   **/
  virtual void malloc(DeviceMemoryPointer * dst) = 0;
  virtual void free(DeviceMemoryPointer * dst) = 0;
  virtual void memcpy(DeviceMemoryPointer * dst, DeviceMemoryPointer * src) = 0;
  virtual void memset(DeviceMemoryPointer * dst, const char value) = 0;

  /**
   * This function implements the following semantic.
   *   for(i=0;i<src.size;i+=src_skip)
   *      func(&dst[f_dst_pos(j)], &src[i])
   * As the name implied, this might be run in parallel.
   *
   * For CPU Device, this could be a simple OpenMP parallel loop.
   * For GPU Device, this could be a kernel that uses func.
   *
   * Strictly speaking, this function has the name `map` mainly because
   * is access pattern and ability to be executed in parallel, instead of
   * its semantic--It is possible to introduce side-effect on `src` and 
   * `f_dst_pos` is not necessarily non-overlapping (it often is, but the 
   * interface does not have a way to enforce it). So maybe a better
   * way of thinking about this function is a parallel for loop with more 
   * structured side-effect (i.e., only on src and dst with a known mapping).
   * 
   **/
  virtual void parallel_map(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    size_t src_skip, FUNC_IDX_MAPPING * f_dst_pos, DeviceMemoryPointer * const f_dst_pos_curry,
    FUNC_MM_MAPPING * func, DeviceMemoryPointer * const func_curry) = 0;

  /**
   * Single-precision operations.
   **/
  virtual void smath_axpy(const float alpha, DeviceMemoryPointer * X, DeviceMemoryPointer * Y) = 0;
  virtual void sapply(DeviceMemoryPointer * dst, FUNC_STRANSFORM * func, DeviceMemoryPointer * const func_curry) = 0;
  virtual void smath_axpby(const float alpha, DeviceMemoryPointer * X, const float beta, DeviceMemoryPointer *Y) = 0;
  virtual void set_num_threads(const int nThreads) = 0;
  virtual void sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
        int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
        float beta, float * pC, int LDC) = 0;

  void selementwise_reduce2(DeviceMemoryPointer *dst, DeviceMemoryPointer *src1, 
    DeviceMemoryPointer *src2, FUNC_SREDUCE * func, DeviceMemoryPointer * const func_curry) ;

  /**
   * Single-precison random number generator.
   **/
  virtual FUNC_STRANSFORM * srand_uni(float, float, DeviceMemoryPointer *) = 0;
  virtual FUNC_STRANSFORM * srand_bern(float, DeviceMemoryPointer *) = 0;
  virtual FUNC_STRANSFORM * srand_gaussian(float, float, DeviceMemoryPointer *) = 0;

  /**
   * Logical functions that only depends on other virtual functions.
   **/
    void sinitialize_xavier(DeviceMemoryPointer *arr, const size_t n_batch) {
      const size_t n_arr_elements = arr->size_in_byte / sizeof(float);
      const size_t fan_in = n_arr_elements / n_batch;
      const float scale = sqrt(3.0 / fan_in);
      DeviceMemoryPointer_Local_RAM generator(NULL, 0);
      auto f_uni = this->srand_uni(-scale, scale, &generator);
      sapply(arr, f_uni, &generator);
    }

   void sbernoulli_initialize(DeviceMemoryPointer *arr, const float p) {
      DeviceMemoryPointer_Local_RAM generator(NULL, 0);
      auto f_bern = this->srand_bern(p, &generator);
      sapply(arr, f_bern, &generator);
    }

    void sgaussian_initialize(DeviceMemoryPointer *arr, const float mean, const float std_dev) {
      DeviceMemoryPointer_Local_RAM generator(NULL, 0);
      auto f_gaussian = this->srand_gaussian(mean, std_dev, &generator);
      sapply(arr, f_gaussian, &generator);
    }

    void sconstant_initialize(DeviceMemoryPointer *arr, const float value) {
      DeviceMemoryPointer_Local_RAM pvalue((void*)&value, sizeof(float));
      sapply(arr, 
        (FUNC_STRANSFORM*)this->choose_ptr((void*)&__sconstant_initialize_helper,
                                            (void*)&_sconstant_initialize_helper),
        &pvalue);
    }

    void smath_apply_grad(DeviceMemoryPointer *X, DeviceMemoryPointer *Y) {
      smath_axpy(-1.0, Y, X);
    }

};

#endif












