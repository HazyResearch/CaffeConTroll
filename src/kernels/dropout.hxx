
#ifndef _KERNEL_DROPOUT_HXX
#define _KERNEL_DROPOUT_HXX

#include "dropout.h"

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline size_t _f_src_to_dst_dropout_forward(size_t src_pos, void * const _arg) {
  return src_pos;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_dropout_forward_train(void * input, void * output, void * const _arg,
    const size_t dst_index) {
  const _dropout_forward_train_arg_helper * const arg = (_dropout_forward_train_arg_helper *) _arg;
  const unsigned int * mask = (unsigned int *) (&arg->mask[dst_index]);
  const float scale = arg->scale;
  const unsigned int threshold = arg->threshold;
  float * const input_data = (float *) input;
  float * const output_data = (float *) output;

  output_data[0] = input_data[0]*(mask[0] > threshold)*scale;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_dropout_forward_test(void * input, void * output, void * const _arg,
    const size_t dst_index) {
  float * const input_data = (float *) input;
  float * const output_data = (float *) output;
  output_data[0] = input_data[0];
}

#endif
