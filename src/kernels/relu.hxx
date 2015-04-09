
#ifndef _KERNEL_RELU_HXX
#define _KERNEL_RELU_HXX

#include "relu.h"

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline size_t _f_src_to_dst_relu_forward(size_t src_pos, void * const _arg) {
  return src_pos;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_relu_forward(void * input, void * output, void * const _arg,
    const size_t dst_index) {
  float * const input_data = (float *) input;
  float * const output_data = (float *) output;

  if (input_data[0] >= 0.) {
    output_data[0] = input_data[0];
  } else {
    output_data[0] = 0.;
  }
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline size_t _f_src_to_dst_relu_backward(size_t src_pos, void * const _arg) {
  return src_pos;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_relu_backward(void * input, void * output, void * const _arg,
    const size_t dst_index) {
  const _relu_backward_arg_helper * const arg = (_relu_backward_arg_helper *) _arg;
  float * const input_grad = (float *) input;
  const float * const output_grad = (float *) output;
  const float * const input_data = (float *) (&arg->input_data[dst_index]);

  input_grad[0] = output_grad[0]*(input_data[0] > 0);
}

#endif
