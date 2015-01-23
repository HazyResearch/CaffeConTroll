//
//  LogicalCube_impl.hxx
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LogicalCube_impl_hxx
#define moka_LogicalCube_impl_hxx

using namespace std;

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

// memcpy doesn't inline with g++, so we use this instead
// (Shamelessly stolen from https://software.intel.com/en-us/articles/memcpy-performance)
inline void * _our_memcpy(void *b, const void *a, size_t n) {
  char *s1 = (char*) b;
  const char *s2 = (const char*)a;
  for(; 0<n; --n)*s1++ = *s2++;
  return b;
}

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
  _our_memcpy(temp_buffer, cube.p_data, sizeof(T)*cube.R*cube.C*cube.B*cube.D);

  size_t dst_index = 0;
  for (size_t c_i = 0; c_i < C; ++c_i) {
    for (size_t r_i = 0; r_i < R; ++r_i) {
      const size_t src_index = c_i*kernel_size + r_i*C*kernel_size;
      _our_memcpy(&cube.p_data[dst_index], &temp_buffer[src_index], sizeof(T)*kernel_size);
      dst_index += kernel_size;
    }
  }

  free(temp_buffer);
}

template <typename T, LayoutType LAYOUT>
template<LoweringType LOWERING>
void LogicalCube<T, LAYOUT>::lower_logical_matrix(const LogicalMatrix<T> * const m,
    const size_t b_i, const size_t d_i, const size_t kernel_size, const size_t stride) {
#ifdef _DO_ASSERT
  assert(stride > 0);
  assert(kernel_size > 0);
  assert(stride < kernel_size);
#endif
  return LoweringHelper<LOWERING>::lower_logical_matrix(*this, m, b_i, d_i, kernel_size, stride);
}

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE1, DUMMY>::lower_logical_matrix(const LogicalCube<T,
    LAYOUT>& cube, const LogicalMatrix<T> * const m, const size_t b_i, const size_t d_i,
    const size_t kernel_size, const size_t stride) {

  const size_t matrix_C = m->C;

  const size_t inverted_kernel_height = m->R - kernel_size + 1;
  const size_t inverted_kernel_width = matrix_C - kernel_size + 1;

  const size_t dst_row_base = d_i*kernel_size*kernel_size;
  const size_t dst_col_base = b_i*inverted_kernel_width*inverted_kernel_width;

  for (size_t i = 0, dst_row_i = dst_row_base, src_i = 0; i < kernel_size;
      i += stride, dst_row_i += kernel_size, src_i += matrix_C) {
    for (size_t j = 0, dst_row = dst_row_i, src_i_j = src_i; j < kernel_size;
        j += stride, ++dst_row, ++src_i_j) {
      //Same as: size_t dst_row = dst_row_base + i*kernel_size + j;

      for (size_t k_r = 0, dst_col = dst_col_base, src = src_i_j; k_r < inverted_kernel_height;
          ++k_r, dst_col += inverted_kernel_width, src += matrix_C) {
        //Same as: size_t dst_col = dst_col_base + k_r*inverted_kernel_width;
        //         size_t src = j + (i + k_r)*m-C;
        _our_memcpy(&cube.p_data[dst_col + dst_row*cube.C], &m->p_data[src],
            inverted_kernel_width*sizeof(T));
      }
    }
  }
}

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE2, DUMMY>::lower_logical_matrix(const LogicalCube<T,
    LAYOUT>& cube, const LogicalMatrix<T> * const m, const size_t b_i, const size_t d_i,
    const size_t kernel_size, const size_t stride) {
  // TODO
}

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE3, DUMMY>::lower_logical_matrix(const LogicalCube<T,
    LAYOUT>& cube, const LogicalMatrix<T> * const m, const size_t b_i, const size_t d_i,
    const size_t kernel_size, const size_t stride) {
  // TODO
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
  p_data(reinterpret_cast<T*>(_p_data)),
  n_elements(_R*_C*_D*_B),
  R(_R), C(_C), D(_D), B(_B),
  own_data(false) {}


template<typename T, LayoutType LAYOUT>
LogicalCube<T, LAYOUT>::LogicalCube(size_t _R, size_t _C, size_t _D, size_t _B) :
  p_data((T*) malloc(sizeof(T)*_R*_C*_D*_B)), // TODO: change to 32byte align
  n_elements(_R*_C*_D*_B),
  R(_R), C(_C), D(_D), B(_B),
  own_data(true) {}

template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::reset_cube() {
  memset(p_data, 0, sizeof(T)*n_elements); // TODO: replace this with our own inline version
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
