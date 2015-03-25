
#ifndef _KERNEL_LRN_HXX
#define _KERNEL_LRN_HXX

#include "lrn.h"

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline size_t _f_src_to_dst_lrn_forward(size_t src_pos, void * const _arg) {
  return 0;
  // const _bias_forward_arg_helper * const arg = (_bias_forward_arg_helper *) _arg;
  // return (src_pos / arg->src_skip) % arg->oD;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_lrn_forward(void * input, void * output, void * const _arg) {
  // const size_t ORxOC = *((size_t *) _arg);
  // const float bias_val = *((float *) bias);
  // float * const output_data = (float *) output;
  // for (size_t i = 0; i < ORxOC; ++i) {
  //   output_data[i] += bias_term;
  // }
}

#endif
