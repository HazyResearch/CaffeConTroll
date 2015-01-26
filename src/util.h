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
#include <float.h>

using std::max;
using std::min;

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

#endif
