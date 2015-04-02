
#ifndef _KERNEL_LOWERING_H
#define _KERNEL_LOWERING_H

#include "../sched/DeviceHeader.h"

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

struct _inverse_lower_cube_arg_helper {
  size_t data_output_width;
  size_t data_output_height;
  size_t kernel_size;
  size_t stride;
  size_t padding;
  size_t iR;
  size_t iC;
  size_t iD;
  size_t iB;
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_inverse_lower_cube(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_inverse_lower_cube(void * input, void * output, void * const _arg, const size_t dst_index);

#endif
