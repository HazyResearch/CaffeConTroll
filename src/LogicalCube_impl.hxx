//
//  LogicalCube_impl.hxx
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LogicalCube_impl_hxx
#define moka_LogicalCube_impl_hxx

#include <type_traits>
#include <string.h>
#include "util.h"

//#include "sched/DeviceDriver_GPU.h"

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
    const size_t B, const size_t kernel_size, DeviceDriver * p_driver) {
  //std::cout << "##############" << std::endl;
  return LoweringHelper<LOWERING>::remap_output(*this, O, B, kernel_size, p_driver);
}


struct _func_src_to_dst_arg_helper{
  size_t kernel_size;
  size_t R;
  size_t C;
  size_t sizeof_T;
};

// TOFIX TO GPU
#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _func_src_to_dst(size_t _dst_index, void * curry){
  const _func_src_to_dst_arg_helper * const parg =
    reinterpret_cast<_func_src_to_dst_arg_helper *>(curry);
  const size_t dst_index = _dst_index/parg->sizeof_T; // TODO, this uglyness implies that DeviceMemoryPointer needs a type.
  const size_t r_i = (dst_index/parg->kernel_size)%parg->R;
  const size_t c_i = (dst_index/parg->kernel_size)/parg->R;
  return (c_i*parg->kernel_size + r_i*parg->C*parg->kernel_size)*parg->sizeof_T;
}

#ifdef _GPU_TARGET
__device__
#endif
FUNC_IDX_MAPPING func_src_to_dst = _func_src_to_dst;

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _sfunc_remap(void * _src, void * _dst, void * curry){
  float * const dst_data = (float *) _dst;
  float * const src_data = (float *) _src;

  const size_t kernel_size = *((size_t*)curry);
  for(size_t i=0;i<kernel_size;i++){
    dst_data[i] = src_data[i];
  }
}

#ifdef _GPU_TARGET
__device__
#endif
// FUNC_MM_MAPPING sfunc_remap = _sfunc_remap;

/*
  auto func_src_to_dst = [=](size_t _dst_index){
      const size_t dst_index = _dst_index/sizeof(T); // TODO, this uglyness implies that DeviceMemoryPointer needs a type.
      const size_t r_i = (dst_index/kernel_size)%R;
      const size_t c_i = (dst_index/kernel_size)/R;
      return (c_i*kernel_size + r_i*C*kernel_size)*sizeof(T);
  };

  auto func_remap = [=](void * _src, void * _dst){
      T * const dst_data = (T *) _dst;
      T * const src_data = (T *) _src;

      for(size_t i=0;i<kernel_size;i++){
        dst_data[i] = src_data[i];
      }
  };
*/

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE1, DUMMY>::remap_output(LogicalCube<T, LAYOUT>& cube, const size_t R, const size_t C,
    const size_t kernel_size, DeviceDriver * p_driver) {

  // TODO: This buffer does not make much sense, but lets get
  // the logical refactoring done first in a way as before
  // before over optimize.
  DeviceMemoryPointer * copy = p_driver->get_device_pointer(NULL, sizeof(T)*cube.R*cube.C*cube.B*cube.D);
  p_driver->malloc(copy);

  //std::cout << "~~~~~~~~~~" <<sizeof(T)*cube.R*cube.C*cube.B*cube.D << std::endl;

  DeviceMemoryPointer * output = cube.get_device_pointer(p_driver);
  p_driver->memcpy(copy, output);

  static_assert(std::is_same<T, float>::value,
            "The func_src_to_dst function needs to change when T <> float.");

  /*
  _func_src_to_dst_arg_helper arg1;
  arg1.kernel_size = kernel_size;
  arg1.R = R;
  arg1.C = C;
  arg1.sizeof_T = sizeof(T);
  DeviceMemoryPointer * parg1 = p_driver->get_device_pointer((void*)&arg1, sizeof(_func_src_to_dst_arg_helper));

  size_t d_kernel = kernel_size;

  DeviceMemoryPointer * parg2 = p_driver->get_device_pointer((void*)&d_kernel, sizeof(size_t));

  p_driver->parallel_map(copy, output, kernel_size*sizeof(T),
      &func_src_to_dst, parg1, &sfunc_remap, parg2);
      */

  PMapHelper args;
  args.dR = cube.R; args.dC = cube.C; args.dD = cube.D; args.dB = cube.B;
  args.sR = cube.R; args.sC = cube.C; args.sD = cube.D; args.sB = cube.B;
  args.dBR = args.dR; args.dBC = args.dC;
  args.sBR = min((size_t)32, args.sR); args.sBC = min((size_t)32, args.sC);

  p_driver->pmap2d_read_coalesce<_fpmap_id,_fmap_remap>(output, copy, args);

  p_driver->free(copy);
  free(copy);
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
LogicalCube<T, LAYOUT>::LogicalCube(size_t _R, size_t _C, size_t _D, size_t _B, DeviceDriver * p_driver) :
  n_elements(_R*_C*_D*_B),
  R(_R), C(_C), D(_D), B(_B),
  own_data(true){
  std::cout << "Allocating " << 1.0*sizeof(T)*_R*_C*_D*_B/1024/1024 << " MB..." << std::endl;
  DeviceMemoryPointer * ptr = p_driver->get_device_pointer(NULL, sizeof(T)*_R*_C*_D*_B);
  p_driver->malloc(ptr);
  p_data = (T*) ptr->ptr;

}



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
