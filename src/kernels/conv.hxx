
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
  // const size_t output_feature_size = oR*oC;
  // const DataType * const bias_data = p_bias_cube->get_p_data();
  // for (size_t o_b = 0; o_b < oB; ++o_b) {
  //   for (size_t o_d = 0; o_d < oD; ++o_d) {
  //     const LogicalMatrix<DataType> output_data_slice =
  //       p_output_layer->p_data_cube->get_logical_matrix(o_d, o_b);
  //     const DataType bias = bias_data[o_d];
  //     for (size_t i = 0; i < output_feature_size; ++i) {
  //       output_data_slice.p_data[i] += bias;
  //     }
  //   }
  // }

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

  // const size_t output_feature_size = oR*oC;
  // p_bias_gradient_cube->reset_cube();
  // DataType * const bias_term = p_bias_gradient_cube->get_p_data();
  // for (size_t o_b = 0; o_b < oB; ++o_b) {
  //   for (size_t o_d = 0; o_d < oD; ++o_d) {
  //     const LogicalMatrix<DataType> input_grad_slice = p_output_layer->p_gradient_cube->get_logical_matrix(o_d, o_b);
  //     DataType sum = DataType(0.0);
  //     for (size_t i = 0; i < output_feature_size; ++i) {
  //       sum += input_grad_slice.p_data[i];
  //     }
  //     //bias_term[o_d] -= stepsize*sum;
  //     bias_term[o_d] += sum;
  //   }
  // }

}

#endif
