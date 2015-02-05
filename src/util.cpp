//
//  util.cpp
//  moka
//
//  Created by Firas Abuzaid on 1/29/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "util.h"

random_device Util::rd;

// We define a explicit specialization just for ints
// (must be defined in the .cpp file; otherwise, we get
// an "explicit specialization" error)
// TODO: figure out why this isn't being used with ints
template <>
inline void Util::constant_initialize<DataType_Int>(DataType_Int * const arr, const DataType_Int value, const size_t n_arr_elements) {
  Util::_our_memset(arr, value, n_arr_elements*sizeof(DataType_Int));
}

// Explicit instantiations for floats and shorts (we don't support strings)
// The implementation is in util.h (weird, but we'll get a linker error otherwise)
template <>
inline void Util::constant_initialize<DataType_SFFloat>(DataType_SFFloat * const arr, const DataType_SFFloat value, const size_t n_arr_elements);

template <>
inline void Util::constant_initialize<DataType_FPFloat>(DataType_FPFloat * const arr, const DataType_FPFloat value, const size_t n_arr_elements);

