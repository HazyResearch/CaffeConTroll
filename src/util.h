//
//  util.h
//  moka
//
//  Created by Firas Abuzaid on 1/25/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_util_h
#define moka_util_h

#include <iostream>
#include <math.h>
#include <random>
#include <float.h>
#include <limits>
#include "cblas.h" // cblas include

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

class Util {
  public:

    // memcpy doesn't inline with clang++/g++, so we use this instead
    // (Shamelessly stolen from https://software.intel.com/en-us/articles/memcpy-performance)
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
      const unsigned char val = (const unsigned char) value;
      for(; 0<n; --n)*s1++ = val;
      return b;
    }

    template <typename T>
    static inline void xavier_initialize(T * const arr, const size_t n_arr_elements, const size_t n_batch) {
      mt19937 gen(rd());
      //mt19937 gen(0); // TODO determinsitic for debugging

      const size_t fan_in = n_arr_elements / n_batch;
      const float scale = sqrt(T(3) / fan_in);
      uniform_real_distribution<T> uni(-scale, scale);
      for (size_t i = 0; i < n_arr_elements; ++i) {
        arr[i] = uni(gen);
      }
    }

    template <typename T>
    static inline void bernoulli_initialize(T * const arr, const size_t n_arr_elements, const float p) {
      mt19937 gen(rd());
      //mt19937 gen(0); // determinsitic for debugging

      bernoulli_distribution bern(p);
      for (size_t i = 0; i < n_arr_elements; ++i) {
        arr[i] = bern(gen);
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

#ifdef _USE_OPENBLAS
    static inline void math_axpy(const int N, const double alpha, const float * X, float * Y)   { cblas_saxpby(N, alpha, X, 1, 1., Y, 1);}
    static inline void math_axpy(const int N, const double alpha, const double * X, double * Y) { cblas_daxpby(N, alpha, X, 1, 1., Y, 1); }
    static inline void set_num_threads(const int nThreads) { openblas_set_num_threads(nThreads); }
#elif _USE_ATLAS
    static inline void math_axpy(const int N, const double alpha, const float * X, float * Y)   { catlas_saxpby(N, alpha, X, 1, 1., Y, 1); }
    static inline void math_axpy(const int N, const double alpha, const double * X, double * Y) { catlas_daxpby(N, alpha, X, 1, 1., Y, 1);}
    static inline void set_num_threads(const int nThreads) {       set_num_threads(nThreads); }

#else
      #error "Select a BLAS framework." 
#endif
    
    // Note: this is only used for shorts and floats, since _our_memset will only work for ints
    template <typename T>
    static inline void constant_initialize(T * const arr, const T value, const size_t n_arr_elements) {
      for(size_t i = 0; i < n_arr_elements; ++i)
        arr[i] = value;
    }

  private:
    static random_device rd;
};

#endif
