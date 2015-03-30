
#ifndef _KERNEL_CONV_HXX
#define _KERNEL_CONV_HXX

#include "conv.h"

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline size_t _f_src_to_dst_bias_forward(size_t src_pos, void * const _arg) {
  const _bias_arg_helper * const arg = (_bias_arg_helper *) _arg;
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

inline size_t _f_src_to_dst_bias_backward(size_t src_pos, void * const _arg) {
  const _bias_arg_helper * const arg = (_bias_arg_helper *) _arg;
  return ((src_pos / arg->src_skip) % arg->oD) * arg->DataTypeSize;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_bias_backward(void * bias, void * input, void * const _arg,
    const size_t dst_index) {
  const size_t ORxOC = *((size_t *) _arg);
  float * const bias_val = (float *) bias;
  const float * const input_grad = (float *) input;
  float sum  = 0.;
  for (size_t i = 0; i < ORxOC; ++i) {
    sum += input_grad[i];
  }
  *bias_val += sum;
}

#endif
