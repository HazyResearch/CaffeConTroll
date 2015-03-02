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

template <typename T, LayoutType LAYOUT>
template<LoweringType LOWERING>
void LogicalCube<T, LAYOUT>::lower_logical_matrix(const LogicalMatrix<T> * const input_matrix,
    const size_t b_i, const size_t d_i, const size_t kernel_size) {
#ifdef _DO_ASSERT
  assert(kernel_size > 0);
#endif
  return LoweringHelper<LOWERING>::lower_logical_matrix(*this, input_matrix, b_i, d_i, kernel_size);
}

template <typename T, LayoutType LAYOUT>
template<LoweringType LOWERING>
void LogicalCube<T, LAYOUT>::lower_logical_matrix(const LogicalMatrix<T> * const input_matrix,
    const size_t b_i, const size_t d_i, const size_t kernel_size, const size_t stride,
    const size_t padding) {
#ifdef _DO_ASSERT
  assert(stride > 0);
  assert(kernel_size > 0);
#endif
  return LoweringHelper<LOWERING>::lower_logical_matrix(*this, input_matrix, b_i, d_i, kernel_size, stride, padding);
}

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE1, DUMMY>::lower_logical_matrix(const LogicalCube<T,
    LAYOUT>& cube, const LogicalMatrix<T> * const input_matrix, const size_t b_i, const size_t d_i,
    const size_t kernel_size) {

  const size_t matrix_C = input_matrix->C;
  const size_t matrix_R = input_matrix->R;
  const T * const input_data = input_matrix->p_data;

  // TODO: if K == 1 or K == N (a fully connected layer)
  // then we should simply copy over the original data into the lowered cube.
  // Right now, though, it doesn't converge when put in this optimization.

  const size_t single_lowering_height = matrix_R - kernel_size + 1;
  const size_t single_lowering_width = matrix_C - kernel_size + 1;

  const size_t dst_row_base = d_i * kernel_size * kernel_size;
  const size_t dst_col_base = b_i * single_lowering_height * single_lowering_width;

  for (size_t src_i = 0, dst_row_i = dst_row_base; src_i < kernel_size * matrix_C;
      dst_row_i += kernel_size, src_i += matrix_C) {
    for (size_t src_i_j = src_i, dst_row = dst_row_i; src_i_j < kernel_size + src_i;
        ++dst_row, ++src_i_j) {
      // Same as: size_t dst_row = dst_row_base + i*kernel_size + j, where 0 <= i < kernel_size
      // and 0 <= j < kernel_size

      for (size_t src = src_i_j, dst_col = dst_col_base; src < src_i_j + single_lowering_height * matrix_C;
          dst_col += single_lowering_width, src += matrix_C) {
        // Same as: size_t dst_col = dst_col_base + k_r*single_lowering_width,
        //          size_t src = j + (i + k_r)*input_matrix->C,
        //          where 0 <= k_r < single_lowering_height
        Util::_our_memcpy(&cube.p_data[dst_col + dst_row*cube.C], &input_data[src],
            single_lowering_width*sizeof(T));
      }
    }
  }
}

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE1, DUMMY>::lower_logical_matrix(const LogicalCube<T,
    LAYOUT>& cube, const LogicalMatrix<T> * const input_matrix, const size_t b_i, const size_t d_i,
    const int kernel_size, const int stride, const int padding) {

  const int height = input_matrix->R;
  const int width = input_matrix->C;
  const T * const input_data = input_matrix->p_data;

  const size_t single_lowering_width = (width + 2 * padding - kernel_size) / stride + 1;
  const size_t single_lowering_height = (height + 2 * padding - kernel_size) / stride + 1;

  const size_t dst_row_base = d_i * kernel_size * kernel_size;
  const size_t dst_col_base = b_i * single_lowering_height * single_lowering_width;

  const int num_height_windows = (height + 2 * padding - kernel_size) / stride + 1; // number of convolution "windows" row-wise
  const int num_width_windows = (width + 2 * padding - kernel_size) / stride + 1; // number of convolution "windows" column-wise

  // i & j keep track of which "window" we're currently calculating. Incremented by 1 each time.
  // src_row_base and src_col_base indicate the starting indices for the window. Incremented by stride each time.
  // dst_col increases every time we calculate a new windo. Incremented by 1 each time.
  for (int src_row_base = -padding, i = 0, dst_col = dst_col_base; i < num_height_windows; src_row_base += stride, ++i) {
    for (int src_col_base = -padding, j = 0; j < num_width_windows; src_col_base += stride, ++dst_col, ++j) {
      // src_row and src_col start at src_row_base and src_col_base, respectively, and iterate a total of kernel_size times. Incremented by 1 each time.
      // dst_row_i starts at dst_row_base. Incremented by kernel_size each time.
      // dst_row starts at dst_row_i. Incremented by 1 each time.
      for (int src_row = src_row_base, dst_row_i = dst_row_base; src_row < kernel_size + src_row_base; ++src_row, dst_row_i += kernel_size) {
        for (int src_col = src_col_base, dst_row = dst_row_i; src_col < kernel_size + src_col_base; ++src_col, ++dst_row) {
          const size_t dst = dst_col + dst_row*cube.C;
          if (src_row < 0 || src_row >= width || src_col < 0 || src_col >= height) {
            cube.p_data[dst] = 0;
          } else {
            cube.p_data[dst] = input_data[src_row*width + src_col];
          }
        }
      }
    }
  }
}

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE2, DUMMY>::lower_logical_matrix(const LogicalCube<T,
    LAYOUT>& cube, const LogicalMatrix<T> * const input_matrix, const size_t b_i, const size_t d_i,
    const size_t kernel_size) {
  // TODO: Lowering type 2, stride == 1, padding == 0
}

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE2, DUMMY>::lower_logical_matrix(const LogicalCube<T,
    LAYOUT>& cube, const LogicalMatrix<T> * const input_matrix, const size_t b_i, const size_t d_i,
    const int kernel_size, const int stride, const int padding) {
  // TODO: Lowering type 2, stride > 1, padding > 0
}

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE3, DUMMY>::lower_logical_matrix(const LogicalCube<T,
    LAYOUT>& cube, const LogicalMatrix<T> * const input_matrix, const size_t b_i, const size_t d_i,
    const size_t kernel_size) {
  // TODO: Lowering type 3, stride == 1, padding == 0
}

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE3, DUMMY>::lower_logical_matrix(const LogicalCube<T,
    LAYOUT>& cube, const LogicalMatrix<T> * const input_matrix, const size_t b_i, const size_t d_i,
    const int kernel_size, const int stride, const int padding) {
  // TODO: Lowering type 3, stride > 1, padding > 0
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
