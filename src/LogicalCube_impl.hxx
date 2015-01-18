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

template<typename T, LayoutType LAYOUT>
LogicalMatrix<T> LogicalCube<T, LAYOUT>::get_logical_matrix(size_t depth_index, size_t batch_index) const {
#ifdef _DO_ASSERT
  assert(depth_index < D); assert(batch_index < B);
#endif
  return LogicalMatrix<T>(&p_data[batch_index*R*C*D + depth_index*R*C], R, C); // Note: for Layout_CRDB only, TODO: support BDRC, the other layout
};


template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::lower_logical_matrix(const LogicalMatrix<T> * const m,
    const size_t b_i, const size_t d_i, const size_t kernel_size, const size_t stride) {

  const size_t inverted_kernel_height = m->R - kernel_size + 1;
  const size_t inverted_kernel_width = m->C - kernel_size + 1;

  for (size_t i = 0; i < kernel_size; i += stride) {
    for (size_t j = 0; j < kernel_size; j += stride) {

      const size_t out_row = d_i*kernel_size*kernel_size + i*kernel_size + j;
      for (size_t k_r = 0; k_r < inverted_kernel_height; ++k_r) {
        const size_t out_col = b_i*inverted_kernel_width*inverted_kernel_width + k_r*inverted_kernel_width;

        memcpy(&p_data[out_col + out_row*C], &m->p_data[j + (i + k_r)*m->C], inverted_kernel_width*sizeof(T));
      }
    }
  }
}

template<typename T, LayoutType LAYOUT>
T * LogicalCube<T, LAYOUT>::logical_get(size_t r, size_t c, size_t d, size_t b) const{
#ifdef _DO_ASSERT
  assert(r<R); assert(c<C); assert(d<D); assert(b<B);
#endif
  return LogicalFetcher<LAYOUT>::logical_get(*this, r,c,d,b);
};

template<typename T, LayoutType LAYOUT>
T * LogicalCube<T, LAYOUT>::physical_get_RCDslice(size_t b){
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
  own_data(false){}


  template<typename T, LayoutType LAYOUT>
  LogicalCube<T, LAYOUT>::LogicalCube(size_t _R, size_t _C, size_t _D, size_t _B) :
    p_data((T*) malloc(sizeof(T)*_R*_C*_D*_B)), // TODO: change to 32byte align
    n_elements(_R*_C*_D*_B),
    R(_R), C(_C), D(_D), B(_B),
    own_data(true){}


    template<typename T, LayoutType LAYOUT>
    LogicalCube<T, LAYOUT>::~LogicalCube(){
      if(own_data){
        free(p_data);
      }
    }

template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::logical_print(){
  for(size_t ib=0;ib<B;ib++){
    for(size_t id=0;id<D;id++){
      cout << "BATCH " << ib << " DEPTH " << id << endl;
      for(size_t ir=0;ir<R;ir++){
        cout << "    " ;
        for(size_t ic=0;ic<C;ic++){
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
void LogicalCube<T, LAYOUT>::physical_print(){
  for(size_t ib=0;ib<B;ib++){
    for(size_t id=0;id<D;id++){
      for(size_t ir=0;ir<R;ir++){
        for(size_t ic=0;ic<C;ic++){
          cout << *logical_get(ir, ic, id, ib) << " ";
        }
      }
    }
  }
  cout << endl;
}

template<typename T, LayoutType LAYOUT>
template<typename TYPECONSTRAINT>
T* LogicalCube<T,LAYOUT>::LogicalFetcher<Layout_CRDB, TYPECONSTRAINT>::logical_get(const LogicalCube<T, LAYOUT>& cube, size_t r, size_t c, size_t d, size_t b){
  //cout << "(" << c + r*cube.C + d*cube.R*cube.C + b*cube.R*cube.C*cube.D << ")" <<
  //__builtin_prefetch((const void*)&cube.p_data[r*cube.C*cube.D*cube.B + c*cube.D*cube.B + d*cube.B + b],0,0);
  return &cube.p_data[c + r*cube.C + d*cube.R*cube.C + b*cube.R*cube.C*cube.D];
}


template<typename T, LayoutType LAYOUT>
template<typename TYPECONSTRAINT>
T* LogicalCube<T,LAYOUT>::LogicalFetcher<Layout_BDRC, TYPECONSTRAINT>::logical_get(const LogicalCube<T, LAYOUT>& cube, size_t r, size_t c, size_t d, size_t b){
  return &cube.p_data[b + d*cube.B + r*cube.B*cube.D + c*cube.B*cube.D*cube.R];
}

template<typename T, LayoutType LAYOUT>
template<typename TYPECONSTRAINT>
T* LogicalCube<T,LAYOUT>::PhysicalFetcher<Layout_CRDB, TYPECONSTRAINT>::physical_get_RCDslice(const LogicalCube<T, LAYOUT>& cube, size_t b){
  return &cube.p_data[b*cube.R*cube.C*cube.D];
}

#endif


