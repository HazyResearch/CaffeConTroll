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

using std::max;
using std::min;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;


#define NOT_IMPLEMENTED std::cerr << "ERROR: Using a bridge with unsupported Layout or DataType." << std::endl; assert(false)

class Util {
  public:
    // TODO: templatize this (right now, causes linker error)
    //template <typename T>
    //static void xavier_initialize(T * const arr, const size_t n_arr_elements, const size_t n_batch);
    static void xavier_initialize(float * const arr, const size_t n_arr_elements, const size_t n_batch);

    //template <typename T>
    static void constant_initialize(float * const arr, const float value, const size_t n_elements);

    static void * _our_memcpy(void *b, const void *a, size_t n);

    static void * _our_memset(void *b, const int value, size_t n);
};

#endif
