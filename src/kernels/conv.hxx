
#ifndef _KERNEL_CONV_HXX
#define _KERNEL_CONV_HXX

#include "conv.h"

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline size_t _f_src_to_dst_bias_forward(size_t src_pos, void * const _arg) {
  const _bias_forward_arg_helper * const arg = (_bias_forward_arg_helper *) _arg;
  return ((src_pos / arg->src_skip) % arg->oD) * arg->DataTypeSize;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_bias_forward(void * bias, void * output, void * const _arg,
    const size_t dst_index) {
  const size_t ORxOC = *((size_t *) _arg);
  const float bias_val = *((float *) bias);
  float * const output_data = (float *) output;
  for (size_t i = 0; i < ORxOC; ++i) {
    output_data[i] += bias_val;
  }
}

#endif
