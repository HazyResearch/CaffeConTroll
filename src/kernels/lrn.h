
#ifndef _KERNEL_LRN_H
#define _KERNEL_LRN_H

#include "../sched/DeviceHeader.h"

struct _lrn_forward_arg_helper {
  int iR;
  int iC;
  int iD;
  int iB;
  int norm_window;
  int local_size;
  char * denoms;
};

struct _lrn_forward_normalize_arg_helper {
  float alpha_over_size;
  float beta;
  char * denoms;
};

struct _lrn_backward_arg_helper {
  int oR;
  int oC;
  int oD;
  int oB;
  float alpha_over_size;
  float beta;
  int norm_window;
  int local_size;
  char * denoms;
  char * input_data;
  char * output_data;
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_lrn_forward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_lrn_forward(void * input, void * output, void * const _arg, const size_t dst_index);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_lrn_forward_normalize(void * input, void * output, void * const _arg, const size_t dst_index);

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_lrn_backward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_lrn_backward(void * input, void * output, void * const _arg, const size_t dst_index);

#endif
