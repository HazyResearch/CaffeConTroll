//
//  LogicalCube_impl.hxx
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LogicalCube_impl_hxx
#define moka_LogicalCube_impl_hxx
#include <string.h>
#include "util.h"

using namespace std;

template<typename T, LayoutType LAYOUT>
T * const LogicalCube<T, LAYOUT>::get_p_data() const {
  return p_data;
}

template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::set_p_data(T * const data) {
#ifdef _DO_ASSERT
  // p_data cannot be updated if this cube
  // owns it data. Otherwise, we will have
  // a memory leak.
  assert(!own_data);
#endif
  p_data = data;
}

/**************************************/
/** Begin code for handling lowering **/
/**************************************/
template<typename T, LayoutType LAYOUT>
LogicalMatrix<T> LogicalCube<T, LAYOUT>::get_logical_matrix(size_t depth_index, size_t batch_index) const {
#ifdef _DO_ASSERT
  assert(depth_index < D); assert(batch_index < B);
#endif
  return LogicalMatrix<T>(&p_data[batch_index*R*C*D + depth_index*R*C], R, C); // Note: for Layout_CRDB only, TODO: support BDRC, the other layout
};

template <typename T, LayoutType LAYOUT>
template<LoweringType LOWERING>
void LogicalCube<T, LAYOUT>::remap_output(const size_t O,
    const size_t B, const size_t kernel_size) {
  return LoweringHelper<LOWERING>::remap_output(*this, O, B, kernel_size);
}

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE1, DUMMY>::remap_output(LogicalCube<T, LAYOUT>& cube, const size_t R, const size_t C,
    const size_t kernel_size) {

  T* temp_buffer = (T*) malloc(sizeof(T)*cube.R*cube.C*cube.B*cube.D);
  Util::_our_memcpy(temp_buffer, cube.p_data, sizeof(T)*cube.R*cube.C*cube.B*cube.D);

  size_t dst_index = 0;
  for (size_t c_i = 0; c_i < C; ++c_i) {
    const size_t src_index_base = c_i*kernel_size;
    for (size_t r_i = 0; r_i < R; ++r_i) {
      const size_t src_index = src_index_base + r_i*C*kernel_size;
      Util::_our_memcpy(&cube.p_data[dst_index], &temp_buffer[src_index], sizeof(T)*kernel_size);
      dst_index += kernel_size;
    }
  }

  free(temp_buffer);
}


/************************************/
/** End code for handling lowering **/
/************************************/

template<typename T, LayoutType LAYOUT>
T * LogicalCube<T, LAYOUT>::logical_get(size_t r, size_t c, size_t d, size_t b) const{
#ifdef _DO_ASSERT
  assert(r<R); assert(c<C); assert(d<D); assert(b<B);
#endif
  return LogicalFetcher<LAYOUT>::logical_get(*this, r,c,d,b);
};

template<typename T, LayoutType LAYOUT>
T * LogicalCube<T, LAYOUT>::physical_get_RCDslice(size_t b) {
#ifdef _DO_ASSERT
  assert(b<B);
#endif
  return PhysicalFetcher<LAYOUT>::physical_get_RCDslice(*this, b);
}

template<typename T, LayoutType LAYOUT>
LogicalCube<T, LAYOUT>::LogicalCube(void * _p_data, size_t _R, size_t _C, size_t _D, size_t _B) :
  n_elements(_R*_C*_D*_B),
  R(_R), C(_C), D(_D), B(_B),
  own_data(false),
  p_data(reinterpret_cast<T*>(_p_data)) {}


template<typename T, LayoutType LAYOUT>
LogicalCube<T, LAYOUT>::LogicalCube(size_t _R, size_t _C, size_t _D, size_t _B) :
  n_elements(_R*_C*_D*_B),
  R(_R), C(_C), D(_D), B(_B),
  own_data(true),
  p_data((T*) malloc(sizeof(T)*_R*_C*_D*_B)) {} // TODO: change to 32byte align

template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::reset_cube() {
  Util::constant_initialize<T>(p_data, T(0.), n_elements);
}

template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::reset_cube(const T val) {
  Util::constant_initialize<T>(p_data, val, n_elements);
}

template<typename T, LayoutType LAYOUT>
LogicalCube<T, LAYOUT>::~LogicalCube() {
  if(own_data) {
    free(p_data);
  }
}

template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::logical_print() const {
  for(size_t ib=0;ib<B;ib++) {
    for(size_t id=0;id<D;id++) {
      cout << "BATCH " << ib << " DEPTH " << id << endl;
      for(size_t ir=0;ir<R;ir++) {
        cout << "    " ;
        for(size_t ic=0;ic<C;ic++) {
          cout << *logical_get(ir, ic, id, ib) << " ";
          //cout << " (" <<
          //(ic + ir*C + id*R*C + ib*R*C*D) << ") ";
        }
        cout << endl;
      }
    }
  }
}

template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::physical_print() const {
  for(size_t ib=0;ib<B;ib++) {
    for(size_t id=0;id<D;id++) {
      for(size_t ir=0;ir<R;ir++) {
        for(size_t ic=0;ic<C;ic++) {
          cout << *logical_get(ir, ic, id, ib) << " ";
        }
      }
    }
  }
  cout << endl;
}

template<typename T, LayoutType LAYOUT>
template<typename TYPECONSTRAINT>
T* LogicalCube<T,LAYOUT>::LogicalFetcher<Layout_CRDB, TYPECONSTRAINT>::logical_get(const LogicalCube<T, LAYOUT>& cube, size_t r, size_t c, size_t d, size_t b) {
  return &cube.p_data[c + r*cube.C + d*cube.R*cube.C + b*cube.R*cube.C*cube.D];
}


template<typename T, LayoutType LAYOUT>
template<typename TYPECONSTRAINT>
T* LogicalCube<T,LAYOUT>::LogicalFetcher<Layout_BDRC, TYPECONSTRAINT>::logical_get(const LogicalCube<T, LAYOUT>& cube, size_t r, size_t c, size_t d, size_t b) {
  return &cube.p_data[b + d*cube.B + r*cube.B*cube.D + c*cube.B*cube.D*cube.R];
}

template<typename T, LayoutType LAYOUT>
template<typename TYPECONSTRAINT>
T* LogicalCube<T,LAYOUT>::PhysicalFetcher<Layout_CRDB, TYPECONSTRAINT>::physical_get_RCDslice(const LogicalCube<T, LAYOUT>& cube, size_t b) {
  return &cube.p_data[b*cube.R*cube.C*cube.D];
}

#endif
