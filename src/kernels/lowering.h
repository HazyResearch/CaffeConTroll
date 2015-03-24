
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

#endif