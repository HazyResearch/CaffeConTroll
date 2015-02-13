//
//  LRNBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/31/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LRNBridge_impl_hxx
#define moka_LRNBridge_impl_hxx

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

  for (size_t i_b = 0; i_b < iB; ++i_b) {
    for (int i_d = 0; i_d < iD; ++i_d) {
      for (size_t i_c = 0; i_c < iC; ++i_c) {
        for (size_t i_r = 0; i_r < iR; ++i_r) {
          DataType sum = DataType(0.);
          for (int i = -norm_window; i <= norm_window; ++i) {
            const int channel = i_d + i;
            if (channel < 0 || channel >= iD) {
              continue; // this means we're in the padding region, so we're adding 0
            }
            sum += pow(*p_input_layer->p_data_cube->logical_get(i_r, i_c, channel, i_b), 2);
          }
          const DataType denom_no_exponent = alpha_over_size * sum + 1;
          const DataType denom = pow(denom_no_exponent, beta);
          *p_output_layer->p_data_cube->logical_get(i_r, i_c, i_d, i_b) =
            *p_input_layer->p_data_cube->logical_get(i_r, i_c, i_d, i_b) / denom;

          *denoms->logical_get(i_r, i_c, i_d, i_b) = denom_no_exponent;
        }
      }
    }
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
