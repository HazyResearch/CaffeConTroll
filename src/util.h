//
//  util.h
//  moka
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_util_h
#define moka_util_h

#include <iostream>
#include <math.h>
#include <random>
#include <float.h>
#include <limits>
#include <assert.h>
#include <boost/random.hpp>

#include "cblas.h"

enum InitializerType {
  CONSTANT = 0,
  BERNOULLI = 1,
  XAVIER = 2,
  GAUSSIAN = 3
};

using std::max;
using std::min;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;
using std::bernoulli_distribution;
using std::normal_distribution;
using std::string;

typedef float  DataType_SFFloat; /*< Single-precision Floating Point. */
typedef short  DataType_FPFloat; /*< 16-bit Fixed Point. */
typedef int    DataType_Int; /*< 32-bit integer. */
typedef string DataType_String; /*< String-type data only for deubgging/unit testing. */

#define NOT_IMPLEMENTED std::cerr << "ERROR: Using a bridge with unsupported Layout or DataType." << std::endl; assert(false)

#ifdef DEBUG
#define _DO_ASSERT
#endif

#ifdef _DETAILED_PROFILING
#define PROFILE_ONLY(x) x
#else
#define PROFILE_ONLY(x)
#endif

class Util {
  public:

    template<typename T>
    static inline void regularize(std::string regularization_type,
        const int n, float lambda, T * const gradient, const T * const current_param) {
      if (regularization_type == "L2") {
        // This can be compiled to SIMD, can optimize later if needed
        for (int i = 0; i < n; ++i) { 
          gradient[i] += lambda * current_param[i];
        }
      }else if (regularization_type == "L1") {
        for (int i = 0; i < n; ++i) {
          gradient[i] += lambda * (current_param[i] > 0 ? 1 : -1);
        }
      }
    }

    /**
     * The following code is from
     * https://github.com/BVLC/caffe/blob/master/src/caffe/solver.cpp#L363
     **/
    static inline float get_learning_rate(std::string lr_policy, float base_lr, float gamma,
      float iter, float caffe_stepsize, float power, float max_iter) {
      float rate, current_step_;
      if (lr_policy == "fixed") {
        rate = base_lr;
      } else if (lr_policy == "step") {
        current_step_ = (int)(iter / caffe_stepsize);
        rate = base_lr * pow(gamma, current_step_);
      } else if (lr_policy == "exp") {
        rate = base_lr * pow(gamma, iter);
      } else if (lr_policy == "inv") {
        rate = base_lr * pow(1.0 + gamma * iter, -power);
      } else if (lr_policy == "multistep") {
        std::cout << "ERROR: We are currently not supporting `multistep` right now." << std::endl;
        assert(false);
      } else if (lr_policy == "poly") {
        rate = base_lr * pow(1.0 - (iter / max_iter), power);
      } else if (lr_policy == "sigmoid") {
        rate = base_lr * (1.0 / (1.0 + exp(-gamma * (iter -caffe_stepsize))));
      } else {
        std::cout << "Unknown learning rate policy: " << lr_policy;
        assert(false);
      }
      return rate;
    }

    // memcpy doesn't inline with clang++/g++, so we use this instead
    // (from https://software.intel.com/en-us/articles/memcpy-performance)
    static inline void * _our_memcpy(void *b, const void *a, size_t n) {
      char *s1 = (char*) b;
      const char *s2 = (const char*)a;
      for(; 0<n; --n)*s1++ = *s2++;
      return b;
    }

    // Same as above: memset doesn't inline with g++, so we use this instead
    static inline void * _our_memset(void *b, const int value, size_t n) {
#ifdef _DO_ASSERT
      assert(value >= -1); // memset will not work correctly if the value is less than -1
#endif
      char * s1 = (char *) b;
      const unsigned char val = (unsigned char) value;
      for(; 0<n; --n)*s1++ = val;
      return b;
    }

    template <typename T>
    static inline void xavier_initialize(T * const arr, const size_t n_arr_elements, const size_t n_batch) {
      mt19937 gen(rd());
      //mt19937 gen(0); // determinsitic for debugging

      const size_t fan_in = n_arr_elements / n_batch;
      const float scale = sqrt(T(3) / fan_in);
      uniform_real_distribution<T> uni(-scale, scale);
      for (size_t i = 0; i < n_arr_elements; ++i) {
        arr[i] = uni(gen);
      }
    }

    template <typename T>
    static inline void bernoulli_initialize(T * const arr, const size_t n_arr_elements, const float p, mt19937 gen) {
      boost::bernoulli_distribution<float> random_distribution(p);
      boost::variate_generator<mt19937, boost::bernoulli_distribution<float> >
          variate_generator(gen, random_distribution);
      for (size_t i = 0; i < n_arr_elements; ++i) {
        arr[i] = variate_generator();
      }
    }

    template <typename T>
    static inline void gaussian_initialize(T * const arr, const size_t n_arr_elements, const T mean, const T std_dev) {
      mt19937 gen(rd());
      //mt19937 gen(0); // determinsitic for debugging

      normal_distribution<T> gaussian(mean, std_dev);
      for (size_t i = 0; i < n_arr_elements; ++i) {
        arr[i] = gaussian(gen);
      }
    }

    // Note: this is only used for shorts and floats, since _our_memset will only work for ints
    template <typename T>
    static inline void constant_initialize(T * const arr, const T value, const size_t n_arr_elements) {
      for (size_t i = 0; i < n_arr_elements; ++i)
        arr[i] = value;
    }

  private:
    static random_device rd;
};

#endif
