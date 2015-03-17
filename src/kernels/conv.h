
#include "../sched/DeviceHeader.h"

#ifndef _KERNEL_CONV_H
#define _KERNEL_CONV_H

struct _bias_forward_arg_helper {
  size_t oR;
  size_t oC;
  size_t oD;
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_bias_forward(size_t a, void * const arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_bias_forward(void * src, void * dst, void * const arg);

#endif
