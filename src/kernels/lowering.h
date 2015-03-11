
#include "../sched/DeviceHeader.h"

#ifndef _KERNEL_LOWERING_H
#define _KERNEL_LOWERING_H

#ifdef _GPU_TARGET
__host__ __device__
#endif 
void _fpmap_id(Block2D * const output_block, const Block2D * const input_block, const PMapHelper * const args);

#ifdef _GPU_TARGET
__host__ __device__
#endif 
void _fmap_lower(float * output, const Block2D * const output_block, const PointIn2DBlock * const input_point, const PMapHelper * const args);

#ifdef _GPU_TARGET
__host__ __device__
#endif 
void _fmap_remap(float * output, const Block2D * const output_block, const PointIn2DBlock * const input_point, const PMapHelper * const args);

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