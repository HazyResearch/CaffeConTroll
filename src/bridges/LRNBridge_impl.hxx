//
//  LRNBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/31/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LRNBridge_impl_hxx
#define moka_LRNBridge_impl_hxx

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



template <typename DataType>
LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::LRNBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_p_input_layer, _p_output_layer, _layer_param),
 alpha(layer_param->lrn_param().alpha()), beta(layer_param->lrn_param().beta()),
 local_size(layer_param->lrn_param().local_size()) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(oR == iR); assert(oC == iC);
  assert(oB == iB); assert(oD == iD);
  assert(alpha >= 0.);
  assert(beta >= 0.);
  assert(local_size % 2 == 1);
#endif

  denoms = new LogicalCube<DataType, Layout_CRDB>(iR, iC, iD, iB);

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Implements LRN in the forward direction. (Note: we only support ACROSS
 * CHANNEL normalization.)
 * This is implemented very differently from Caffe, but it should still
 * produce the same result.
 **/
template <typename DataType>
void LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::forward() {
  report_forward_last_transfer.reset();

  p_input_layer->p_gradient_cube->reset_cube();
  
  const DataType alpha_over_size = alpha / local_size;

  const int norm_window = (int) local_size / 2;

  const int _iB = iB;
  const int _iD = iD;
  const int _iC = iC;
  const int _iR = iR;

  const int iRiC = iR*iC;

  DataType sum;
  DataType num;
  DataType denom_no_exponent;
  DataType denom;

  DataType * p_denoms = denoms->p_data;
  DataType * p_output = p_output_layer->p_data_cube->p_data;
  DataType * p_input, * p_input_toremove, * p_input_toadd, * p_input_tmp, *p_input_all;
  p_input_all = p_input_layer->p_data_cube->p_data;

  // first, fill in p_denoms with the sum
  for (int i_b = 0; i_b < _iB; ++i_b) {

    // do the first depth in the old way
    p_input = p_input_layer->p_data_cube->logical_get(0, 0, -norm_window, i_b);
    for (int iric = 0; iric < iRiC; iric ++) {
      DataType sum = DataType(0.);
      p_input_tmp = p_input;
      for (int i = -norm_window; i <= norm_window; ++i) {
        sum += (i < 0 || i >= iD) ? 0 : (*p_input_tmp)*(*p_input_tmp);
        p_input_tmp += iRiC;
      }
      *p_denoms = sum;
      p_input ++;
      p_denoms ++;
    }

    // for other batch, reuse the old result
    for (int i_d = 1; i_d < _iD; ++i_d) {

      DataType * p_denoms2 = p_denoms;
      for (int iric = 0; iric < iRiC; iric ++) {
        *(p_denoms2) = *(p_denoms2 - iRiC);
        p_denoms2 ++;
      }

      p_denoms2 = p_denoms;
      if(i_d-norm_window-1 >= 0){
        p_input_toremove = p_input_layer->p_data_cube->logical_get(0, 0, i_d-norm_window-1, i_b);
        for (int iric = 0; iric < iRiC; iric ++) {
          *p_denoms2 -= (*p_input_toremove) * (*p_input_toremove);
          p_input_toremove ++;
          p_denoms2 ++;
        }
      }

      p_denoms2 = p_denoms;
      if(i_d+norm_window < _iD){
        p_input_toadd = p_input_layer->p_data_cube->logical_get(0, 0, i_d+norm_window, i_b);
        for (int iric = 0; iric < iRiC; iric ++) {
          *p_denoms2 += (*p_input_toadd) * (*p_input_toadd);
          p_input_toadd ++;
          p_denoms2 ++;
        }
      }

      p_denoms += iRiC;
    }

  }

  // then do normalization
  p_denoms = denoms->p_data;
  p_output = p_output_layer->p_data_cube->p_data;
  p_input_layer->p_data_cube->p_data;
  const size_t n_elements = _iB*_iD*_iC*_iR;
  for (size_t i = 0; i < n_elements; ++i) {
    *p_denoms = alpha_over_size*(*p_denoms) + 1;
    *p_output = (*p_input_all)/fastPrecisePow(*p_denoms, beta);
    p_output ++;
    p_denoms ++;
    p_input_all ++;
  }

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}

/**
 * Implements LRN in the backward direction. (Note: we only support ACROSS
 * CHANNEL normalization.)
 * This is also implemented very differently from Caffe, but it should still
 * produce the same result.
 **/
template <typename DataType>
void LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::backward() {
  report_backward_updateweight_last_transfer.reset();

  const DataType alpha_over_size = alpha / local_size;
  const int norm_window = (int) local_size / 2;

  for (size_t o_b = 0; o_b < iB; ++o_b) {
    for (int o_d = 0; o_d < iD; ++o_d) {
      for (size_t o_c = 0; o_c < iC; ++o_c) {
        for (size_t o_r = 0; o_r < iR; ++o_r) {
          const DataType denom_no_exponent = *denoms->logical_get(o_r, o_c, o_d, o_b);
          const DataType denom = pow(denom_no_exponent, beta);
          const DataType input_data = *p_input_layer->p_data_cube->logical_get(o_r, o_c, o_d, o_b);
          for (int i = -norm_window; i <= norm_window; ++i) {
            const int channel = o_d + i;
            if (channel < 0 || channel >= iD) {
              continue; // in the padding region, so we're adding 0
            }
            const DataType output_grad = *p_output_layer->p_gradient_cube->logical_get(o_r, o_c, channel, o_b);
            const DataType window_data = *p_input_layer->p_data_cube->logical_get(o_r, o_c, channel, o_b);
            DataType input_grad;
            if(i == 0)
              input_grad = (denom - 2 * input_data * window_data * beta * alpha_over_size * pow(denom_no_exponent, beta - 1)) / pow(denom, 2);
            else
              input_grad = (- 2 * input_data * window_data * beta * alpha_over_size * pow(denom_no_exponent, beta - 1)) / pow(denom, 2);

            *p_input_layer->p_gradient_cube->logical_get(o_r, o_c, o_d, o_b) += input_grad * output_grad;
          }
        }
      }
    }
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType>
LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::~LRNBridge() {
  delete denoms;
}

#endif
