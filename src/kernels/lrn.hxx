
#ifndef _KERNEL_LRN_HXX
#define _KERNEL_LRN_HXX

#include "lrn.h"

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline size_t _f_src_to_dst_lrn_forward(size_t src_pos, void * const _arg) {
  return src_pos;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_lrn_forward(void * input, void * output, void * const _arg,
    const size_t dst_index) {
  const _lrn_forward_arg_helper * const arg = (_lrn_forward_arg_helper *) _arg;
  const int iR = arg->iR;
  const int iC = arg->iC;
  const int iD = arg->iD;
  const int norm_window = arg->norm_window;
  float * denoms = (float *) (&arg->denoms[dst_index]);
  float * const input_data = (float *) input;

  const int iRiC = iR*iC;
  float * denoms_tmp, * input_data_tmp, * input_data_tmp2,
        * input_data_to_remove, * input_data_to_add;
  // first, fill in denoms with the sum
  // do the first depth in the old way
  input_data_tmp = input_data;
  for (int ir_ic = 0; ir_ic < iRiC; ir_ic++) {
    float sum = 0.;
    input_data_tmp2 = input_data_tmp;
    for (int i = 0; i <= norm_window; ++i) {
      sum += (i < 0 || i >= iD) ? 0 : (*input_data_tmp2)*(*input_data_tmp2);
      input_data_tmp2 += iRiC;
    }
    *denoms = sum;
    input_data_tmp++;
    denoms++;
  }

  // for the rest of the depths, reuse the old result
  for (int i_d = 1; i_d < iD; ++i_d) {
    denoms_tmp = denoms;
    for (int ir_ic = 0; ir_ic < iRiC; ir_ic++) {
      *(denoms_tmp) = *(denoms_tmp - iRiC);
      denoms_tmp++;
    }

    denoms_tmp = denoms;
    if (i_d - norm_window - 1 >= 0) {
      input_data_to_remove = &input_data[(i_d - norm_window - 1)*iRiC];
      // input_data_to_remove = input_data_layer->p_data_cube->logical_get(0, 0, i_d - norm_window - 1, i_b);
      for (int ir_ic = 0; ir_ic < iRiC; ir_ic++) {
        *denoms_tmp -= (*input_data_to_remove)*(*input_data_to_remove);
        input_data_to_remove++;
        denoms_tmp++;
      }
    }

    denoms_tmp = denoms;
    if (i_d + norm_window < iD) {
      input_data_to_add = &input_data[(i_d + norm_window)*iRiC];
      // input_data_to_add = input_data_layer->p_data_cube->logical_get(0, 0, i_d + norm_window, i_b);
      for (int ir_ic = 0; ir_ic < iRiC; ir_ic++) {
        *denoms_tmp += (*input_data_to_add)*(*input_data_to_add);
        input_data_to_add++;
        denoms_tmp++;
      }
    }
    denoms += iRiC;
  }
}

#endif
