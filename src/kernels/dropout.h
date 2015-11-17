
#ifndef _KERNEL_DROPOUT_H
#define _KERNEL_DROPOUT_H

#include "../sched/DeviceHeader.h"

struct _dropout_forward_train_arg_helper {
  char * mask;
  float scale;
  unsigned int threshold;
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_dropout_forward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_dropout_forward_train(void * input, void * output, void * const _arg, const size_t dst_index);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_dropout_forward_test(void * input, void * output, void * const _arg, const size_t dst_index);

#endif
