//
//  LogicalCube_impl.hxx
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LogicalCube_impl_hxx
#define moka_LogicalCube_impl_hxx

#include <type_traits>
#include <string.h>
#include "util.h"

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

template<typename T, LayoutType LAYOUT>
template<typename DUMMY>
void LogicalCube<T, LAYOUT>::LoweringHelper<LOWERING_TYPE1, DUMMY>::remap_output(LogicalCube<T, LAYOUT>& cube, const size_t R, const size_t C,
    const size_t kernel_size, DeviceDriver * p_driver) {

  // SHADJIS TODO: If this ever fails just remove the assert, but I think this is never needed
  assert(false);
  
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
  p_data(reinterpret_cast<T*>(_p_data)),
  p_data_device_ptr(NULL) {}

// SHADJIS TODO: Why does this malloc and not new? Why not call devicemalloc?
// Why does this constructor even exist? What is the driver?
// Need to change this to only have a single constructor in which cube owns data,
// with the driver required
template<typename T, LayoutType LAYOUT>
LogicalCube<T, LAYOUT>::LogicalCube(size_t _R, size_t _C, size_t _D, size_t _B) :
  n_elements(_R*_C*_D*_B),
  R(_R), C(_C), D(_D), B(_B),
  own_data(true),
  p_data((T*) malloc(sizeof(T)*_R*_C*_D*_B)),
  p_data_device_ptr(NULL),
  p_data_driver(NULL) {} // TODO: change to 32byte align

template<typename T, LayoutType LAYOUT>
LogicalCube<T, LAYOUT>::LogicalCube(size_t _R, size_t _C, size_t _D, size_t _B, DeviceDriver * p_driver) :
  n_elements(_R*_C*_D*_B),
  R(_R), C(_C), D(_D), B(_B),
  own_data(true),
  p_data_device_ptr(NULL) {

//#ifdef _DETAILED_PROFILING
//  std::cout << "Allocating " << 1.0*sizeof(T)*_R*_C*_D*_B/1024/1024 << " MB..." << std::endl;
//#endif
  // Allocate a device memory pointer (SHADJIS TODO: small memory leak)
  p_data_device_ptr = p_driver->get_device_pointer(NULL, sizeof(T)*_R*_C*_D*_B);
  // Now use the new device memory pointer to call a malloc on the device
  p_driver->malloc(p_data_device_ptr);
  p_data = (T*) p_data_device_ptr->ptr;
  p_data_driver = p_driver; // SHADJIS TODO: What if this is deleted? Should make copy for this class?
}
template<typename T, LayoutType LAYOUT>
LogicalCube<T, LAYOUT>::LogicalCube(void * _p_data, size_t _R, size_t _C, size_t _D, size_t _B,
    DeviceDriver * p_driver) :
  n_elements(_R*_C*_D*_B),
  R(_R), C(_C), D(_D), B(_B),
  own_data(false),
  p_data(reinterpret_cast<T*>(_p_data)),
  p_data_device_ptr(NULL) {}

template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::reset_cube() {
  Util::constant_initialize<T>(p_data, T(0.), n_elements);
}

template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::reset_cube(const T val) {
  Util::constant_initialize<T>(p_data, val, n_elements);
}

// SHADJIS TODO: This should only call p_data_driver->free, not free,
// but we have a special-case constructor which takes no driver and
// calls malloc. Should make that one use a default CPU driver (so
// there is always some driver) and never just call free.
template<typename T, LayoutType LAYOUT>
LogicalCube<T, LAYOUT>::~LogicalCube() {
  if(own_data) {
    if (p_data_device_ptr) { // Is non-NULL if we called the p_driver constructor
	  p_data_driver->free(p_data_device_ptr);
	} else {
      free(p_data);
	}
  }
}

template<typename T, LayoutType LAYOUT>
void LogicalCube<T, LAYOUT>::logical_print() const {
  for(size_t ib=0;ib<B;ib++) {
    for(size_t id=0;id<D;id++) {
      std::cout << "BATCH " << ib << " DEPTH " << id << std::endl;
      for(size_t ir=0;ir<R;ir++) {
        std::cout << "    " ;
        for(size_t ic=0;ic<C;ic++) {
          std::cout << *logical_get(ir, ic, id, ib) << " ";
          //std::cout << " (" <<
          //(ic + ir*C + id*R*C + ib*R*C*D) << ") ";
        }
        std::cout << std::endl;
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
          std::cout << *logical_get(ir, ic, id, ib) << " ";
        }
      }
    }
  }
  std::cout << std::endl;
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
