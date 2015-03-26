
#ifndef _KERNEL_SOFTMAX_H
#define _KERNEL_SOFTMAX_H

#include "../sched/DeviceHeader.h"

struct _softmax_forward_arg_helper {
  float * loss;
  size_t iR;
  size_t iC;
  size_t iD;
  char * ground_truth;
};

struct _softmax_backward_arg_helper {
  size_t iR;
  size_t iC;
  size_t iD;
  size_t iB;
  char * ground_truth;
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_softmax_forward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_softmax_forward(void * output, void * input, void * const _arg, const size_t dst_index);

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_softmax_backward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_softmax_backward(void * output, void * input, void * const _arg, const size_t dst_index);

#endif
