
#include "mul.h"

#ifndef _KERNEL_MUL_HXX
#define _KERNEL_MUL_HXX

#ifdef _GPU_TARGET
__host__ __device__ 
#endif
inline float _f_reduce_mul(float a, float b, void * const){
  return a*b;
}


#ifdef _GPU_TARGET
__host__ __device__ 
#endif
inline float _f_reduce_tanhgrad(float a, float b, void * const){
  return 1-(a*a)*b;
}

#endif




