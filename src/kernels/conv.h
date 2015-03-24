
#ifndef _KERNEL_CONV_H
#define _KERNEL_CONV_H

#include "../sched/DeviceHeader.h"

struct _bias_forward_arg_helper {
  size_t src_skip;
  size_t DataTypeSize;
  size_t oD;
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_bias_forward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_bias_forward(void * bias, void * output, void * const _arg, const size_t dst_index);

#endif
