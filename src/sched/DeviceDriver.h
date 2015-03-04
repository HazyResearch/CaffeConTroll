
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

  virtual DeviceMemoryPointer * get_device_pointer(void * ptr, size_t size_in_byte) = 0;

  /**
   * Memory manipulation and data movement.
   **/
  virtual void memcpy(DeviceMemoryPointer dst, DeviceMemoryPointer src) = 0;
  virtual void memset(DeviceMemoryPointer dst, const char value) = 0;

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
  virtual void parallel_map(DeviceMemoryPointer dst, DeviceMemoryPointer src, 
    size_t src_skip, std::function<size_t(size_t)> f_dst_pos,
    std::function<void(void *, void *)> func) = 0;

  /**
   * Single-precision operations.
   **/
  virtual void smath_axpy(const float alpha, DeviceMemoryPointer X, DeviceMemoryPointer Y) = 0;
  virtual void sapply(DeviceMemoryPointer dst, size_t n_element, std::function<void(float&)> func) = 0;
  virtual void smath_axpby(const float alpha, DeviceMemoryPointer X, const float beta, DeviceMemoryPointer Y) = 0;
  virtual void set_num_threads(const int nThreads) = 0;
  virtual void sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
        int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
        float beta, float * pC, int LDC) = 0;

  void selementwise_reduce2(DeviceMemoryPointer dst, DeviceMemoryPointer src1, 
    DeviceMemoryPointer src2, std::function<float(float,float)> FUNC) ;

  /**
   * Single-precison random number generator.
   **/
  virtual std::function<void(float&)> srand_uni(float, float) = 0;
  virtual std::function<void(float&)> srand_bern(float) = 0;
  virtual std::function<void(float&)> srand_gaussian(float, float) = 0;

  /**
   * Logical functions that only depends on other virtual functions.
   **/
    void sinitialize_xavier(DeviceMemoryPointer arr, const size_t n_batch) {
      const size_t n_arr_elements = arr.size_in_byte / sizeof(float);
      const size_t fan_in = n_arr_elements / n_batch;
      const float scale = sqrt(3.0 / fan_in);
      auto f_uni = this->srand_uni(-scale, scale);
      sapply(arr, n_arr_elements, f_uni);
    }

   void sbernoulli_initialize(DeviceMemoryPointer arr, const float p) {
      const size_t n_arr_elements = arr.size_in_byte / sizeof(float);
      auto f_bern = this->srand_bern(p);
      sapply(arr, n_arr_elements, f_bern);
    }

    void sgaussian_initialize(DeviceMemoryPointer arr, const float mean, const float std_dev) {
      const size_t n_arr_elements = arr.size_in_byte / sizeof(float);
      auto f_gaussian = this->srand_gaussian(mean, std_dev);
      sapply(arr, n_arr_elements, f_gaussian);
    }

    void sconstant_initialize(DeviceMemoryPointer arr, const float value) {
      const size_t n_arr_elements = arr.size_in_byte / sizeof(float);
      auto f_set_to_const = [=](float & b) { b = value; };
      sapply(arr, n_arr_elements, f_set_to_const);
    }

    void smath_apply_grad(DeviceMemoryPointer X, DeviceMemoryPointer Y) {
      smath_axpy(-1.0, Y, X);
    }

};

#endif












