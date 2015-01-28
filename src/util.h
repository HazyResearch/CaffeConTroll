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

using std::max;
using std::min;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;

random_device rd;
mt19937 gen(rd());

#define NOT_IMPLEMENTED std::cerr << "ERROR: Using a bridge with unsupported Layout or DataType." << std::endl; assert(false)

// memcpy doesn't inline with g++, so we use this instead
// (Shamelessly stolen from https://software.intel.com/en-us/articles/memcpy-performance)
inline void * _our_memcpy(void *b, const void *a, size_t n) {
  char *s1 = (char*) b;
  const char *s2 = (const char*)a;
  for(; 0<n; --n)*s1++ = *s2++;
  return b;
}

// Same as above: memset doesn't inline with g++, so we use this instead
inline void * _our_memset(void *b, const int value, size_t n) {
  char *s1 = (char*) b;
  const unsigned char val = (const unsigned char) value;
  for(; 0<n; --n)*s1++ = val;
  return b;
}

template <typename T>
inline void xavier_initialize(T * const arr, const size_t n_arr_elements, const size_t n_batch) {
  const size_t fan_in = n_arr_elements / n_batch;
  const T scale = sqrt(T(3) / fan_in);
  uniform_real_distribution<T> uni(-scale, scale);
  for (size_t i = 0; i < n_arr_elements; ++i) {
    arr[i] = uni(gen);
  }

}

template <typename T>
inline void constant_initialize(T * const arr, const T value, const size_t n_elements) {
  _our_memset(arr, value, n_elements*sizeof(T));
}


#endif
