
#include "../sched/DeviceHeader.h"

#ifndef _KERNEL_TEST_H
#define _KERNEL_TEST_H

#ifdef _GPU_TARGET
__host__ __device__ 
#endif
float _f_add_one(float a, void * const arg);

#ifdef _GPU_TARGET
__host__ __device__ 
#endif
float _f_set(float a, void * const arg);

#ifdef _GPU_TARGET
__host__ __device__ 
#endif
float _f_reduce(float a, float b, void * const arg);

#ifdef _GPU_TARGET
__host__ __device__ 
#endif
size_t _f_idx_strid4_copy(size_t a, void * const arg);

#ifdef _GPU_TARGET
__host__ __device__ 
#endif
void _f_strid4_copy(void * dst, void * src, void * const arg);


#endif