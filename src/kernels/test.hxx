#include "test.h"

#ifndef _KERNEL_TEST_HXX
#define _KERNEL_TEST_HXX


#ifdef _GPU_TARGET
__host__ __device__ 
#endif
inline float _f_add_one(float a, void * const arg){
  return a + *((float *) arg);
}

#ifdef _GPU_TARGET
__host__ __device__ 
#endif
inline float _f_set(float a, void * const arg){
  return *((float *) arg);
}

#ifdef _GPU_TARGET
__host__ __device__ 
#endif
inline float _f_reduce(float a, float b, void * const arg){
	return a + b + *((float *) arg);
}

#ifdef _GPU_TARGET
__host__ __device__ 
#endif
inline size_t _f_idx_strid4_copy(size_t a, void * const arg){
	return a;
}

#ifdef _GPU_TARGET
__host__ __device__ 
#endif
inline void _f_strid4_copy(void * dst, void * src, void * const arg){
	float * const _dst = (float *) dst;
	float * const _src = (float *) src;
	for(int i=0;i<4;i++){
		_dst[i] = _src[i] + *((float *) arg);
	}
}


#endif




