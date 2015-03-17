#include "conv.h"

#ifndef _KERNEL_CONV_HXX
#define _KERNEL_CONV_HXX

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline size_t _f_src_to_dst_bias_forward(size_t a, void * const arg) {
  // TODO
  return 0;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
// NOTE: Normally the first argument should be dst, and the second should be
// src. Here, we reverse the two, because we want the bias argument to correspond
// correspond to src and the output argument to correspond to dst.
inline void _f_bias_forward(void * src, void * dst, void * const arg) {
  // TODO
}

#endif
