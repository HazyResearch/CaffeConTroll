
#ifndef _KERNEL_SOFTMAX_HXX
#define _KERNEL_SOFTMAX_HXX

#include "softmax.h"

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline size_t _f_src_to_dst_softmax_forward(size_t src_pos, void * const _arg) {
  const _softmax_forward_arg_helper * const arg = (_softmax_forward_arg_helper *) _arg;
  return src_pos / arg->iR / arg->iC;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_softmax_forward(void * output, void * input, void * const _arg,
    const size_t dst_index) {
  const _softmax_forward_arg_helper * const arg = (_softmax_forward_arg_helper *) _arg;
  float * const loss = arg->loss;
  const size_t iR = arg->iR;
  const size_t iC = arg->iC;
  const size_t iD = arg->iD;
  float * ground_truth = (float *) (&arg->ground_truth[dst_index / arg->iR / arg->iC / arg->iD]);

  const float * const single_input_batch = (float *) input;
  float * const output_data = (float *) output;

  float max = single_input_batch[0];
  const size_t size_of_single_batch = iR*iC*iD;
  for (size_t i = 1; i < size_of_single_batch; ++i) {
    if (single_input_batch[i] > max) {
      max = single_input_batch[i];
    }
  }

  float denom = 0.;
  for (size_t i = 0; i < size_of_single_batch; ++i) {
    const float exponentiated_val = exp(single_input_batch[i] - max);
    denom += exponentiated_val;
  }

  for (size_t i = 0; i < size_of_single_batch; ++i) {
    output_data[i] = exp(single_input_batch[i] - max) / denom;
  }

  *loss -= log(output_data[static_cast<int>(ground_truth[0])]);
}

#endif
