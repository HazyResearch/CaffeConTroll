
#ifndef _KERNEL_LRN_HXX
#define _KERNEL_LRN_HXX

#include "lrn.h"

/**
 * This function is from
 * http://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp/
 **/
inline double fastPrecisePow(double a, double b) {
  int e = (int) b;
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;

  double r = 1.0;
  while (e) {
    if (e & 1) {
      r *= a;
    }
    a *= a;
    e >>= 1;
  }
  return r * u.d;
}

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
      for (int ir_ic = 0; ir_ic < iRiC; ir_ic++) {
        *denoms_tmp -= (*input_data_to_remove)*(*input_data_to_remove);
        input_data_to_remove++;
        denoms_tmp++;
      }
    }

    denoms_tmp = denoms;
    if (i_d + norm_window < iD) {
      input_data_to_add = &input_data[(i_d + norm_window)*iRiC];
      for (int ir_ic = 0; ir_ic < iRiC; ir_ic++) {
        *denoms_tmp += (*input_data_to_add)*(*input_data_to_add);
        input_data_to_add++;
        denoms_tmp++;
      }
    }
    denoms += iRiC;
  }
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_lrn_forward_normalize(void * input, void * output, void * const _arg,
    const size_t dst_index) {
  const _lrn_forward_normalize_arg_helper * const arg = (_lrn_forward_normalize_arg_helper *) _arg;
  const float alpha_over_size = arg->alpha_over_size;
  const float beta = arg->beta;

  const float * const input_data = (float *) input;
  float * const denoms = (float *) (&arg->denoms[dst_index]);
  float * const output_data = (float *) output;

  denoms[0] = alpha_over_size*(denoms[0]) + 1;
#ifdef _FASTPOW
  output_data[0] = (input_data[0])/fastPrecisePow(denoms[0], beta);
#else
  output_data[0] = (input_data[0])/pow(denoms[0], beta);
#endif
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline size_t _f_src_to_dst_lrn_backward(size_t src_pos, void * const _arg) {
  return src_pos;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _f_lrn_backward(void * input, void * output, void * const _arg,
    const size_t dst_index) {
  const _lrn_backward_arg_helper * const arg = (_lrn_backward_arg_helper *) _arg;
  const int oR = arg->oR;
  const int oC = arg->oC;
  const int oD = arg->oD;
  const float alpha_over_size = arg->alpha_over_size;
  const float beta = arg->beta;
  const int norm_window = arg->norm_window;

  const float * const denoms = (float *) (&arg->denoms[dst_index]);
  const float * const input_data = (float *) (&arg->input_data[dst_index]);
  const float * const output_grad = (float *) output;
  float * const input_grad = (float *) input;

  for (int o_d = 0; o_d < oD; ++o_d) {
    for (size_t o_r = 0; o_r < oR; ++o_r) {
      for (size_t o_c = 0; o_c < oC; ++o_c) {
        const float denom_no_exponent = denoms[o_c + o_r*oC + o_d*oR*oC];
#ifdef _FASTPOW
        const float denom = fastPrecisePow(denom_no_exponent, beta);
#else
        const float denom = pow(denom_no_exponent, beta);
#endif
        const float denom_n1 = 1./(denom*denom_no_exponent);
        const float output_grad_val = output_grad[o_c + o_r*oC + o_d*oR*oC];
        const float window_data = input_data[o_c + o_r*oC + o_d*oR*oC];

        float input_grad_val;
        float input_grad_val2;
        input_grad_val2 = beta*denom_n1*alpha_over_size*2*window_data;

        for (int i = -norm_window; i <= norm_window; ++i) {
          const int channel = o_d + i;
          if (channel < 0 || channel >= oD) {
            continue; // in the padding region, so we're adding 0
          }
          const float input_data_val = input_data[o_c + o_r*oC + channel*oR*oC];

          if (i == 0) {
            input_grad_val = 1./denom - input_grad_val2*input_data_val;
          } else {
            input_grad_val = -input_grad_val2*input_data_val;
          }
          input_grad[o_c + o_r*oC + channel*oR*oC] += input_grad_val*output_grad_val;
        }
      }
    }
  }
}

#endif
