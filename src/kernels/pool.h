
#ifndef _KERNEL_POOL_H
#define _KERNEL_POOL_H

#include "../sched/DeviceHeader.h"

struct _pool_forward_arg_helper {
  int pooled_height;
  int pooled_width;
  int stride;
  int kernel_size;
  int iR;
  int iC;
  int oR;
  int oC;
  char * max_index;
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_pool_forward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_pool_forward(void * output, void * input, void * const _arg, const size_t dst_index);

#endif
