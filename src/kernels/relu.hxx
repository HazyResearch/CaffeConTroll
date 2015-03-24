
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
  output_data[0] = std::max<float>(input_data[0], 0.);
}

#endif
