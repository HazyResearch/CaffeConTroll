//
//  util.cpp
//  moka
//
//  Created by Firas Abuzaid on 1/29/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "util.h"

//template <typename T>
//void Util::xavier_initialize(T * const arr, const size_t n_arr_elements, const size_t n_batch) {
void Util::xavier_initialize(float * const arr, const size_t n_arr_elements, const size_t n_batch) {
  random_device rd;
  mt19937 gen(rd());

  const size_t fan_in = n_arr_elements / n_batch;
  const float scale = sqrt(/*T(3)*/3.0 / fan_in);
  uniform_real_distribution<float> uni(-scale, scale);
  for (size_t i = 0; i < n_arr_elements; ++i) {
    arr[i] = uni(gen);
  }
}

//template <typename T>
void Util::constant_initialize(float * const arr, const float value, const size_t n_elements) {
  Util::_our_memset(arr, value, n_elements*sizeof(float));
}

/* TODO: inline these */

// memcpy doesn't inline with g++, so we use this instead
// (Shamelessly stolen from https://software.intel.com/en-us/articles/memcpy-performance)
void * Util::_our_memcpy(void *b, const void *a, size_t n) {
  char *s1 = (char*) b;
  const char *s2 = (const char*)a;
  for(; 0<n; --n)*s1++ = *s2++;
  return b;
}

// Same as above: memset doesn't inline with g++, so we use this instead
void * Util::_our_memset(void *b, const int value, size_t n) {
  char *s1 = (char*) b;
  const unsigned char val = (const unsigned char) value;
  for(; 0<n; --n)*s1++ = val;
  return b;
}

