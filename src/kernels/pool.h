
#ifndef _KERNEL_POOL_H
#define _KERNEL_POOL_H

#include "../sched/DeviceHeader.h"

struct _pool_forward_arg_helper {
  int stride;
  int kernel_size;
  int iR;
  int iC;
  int oR;
  int oC;
  int D;
  int B;
  int *max_index; // SHADJIS TODO: unsigned char?
};

struct _pool_backward_arg_helper {
  int stride;       // Just for GPU
  int kernel_size;  // Just for GPU
  int iR;
  int iC;
  int oR;
  int oC;
  int D;
  int B;
  int * max_index; // SHADJIS TODO: unsigned char?
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_pool_forward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_pool_forward(void * output, void * input, void * const _arg, const size_t dst_index);

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_pool_backward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_pool_backward(void * output, void * input, void * const _arg, const size_t dst_index);

#endif
