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
LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::LRNBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
    const float _alpha, const float _beta, const size_t _local_size)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_p_input_layer, _p_output_layer),
 alpha(_alpha),
 beta(_beta),
 local_size(_local_size) {
  this->report_forward_constructor.reset();
  this->report_forward_last_transfer.reset();
  this->report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(oR == iR); assert(oC == iC);
  assert(oB == iB); assert(oD == iD);
  assert(alpha >= 0.);
  assert(beta >= 0.);
  assert(local_size % 2 == 1);
#endif

  denoms = new LogicalCube<DataType, Layout_CRDB>(iR, iC, iD, iB);

  this->report_forward_constructor.end(0, 0, 0);
}

/**
 * Implements LRN in the forward direction. (Note: we only support ACROSS
 * CHANNEL normalization.)
 * This is implemented very differently from Caffe, but it should still
 * produce the same result.
 **/
template <typename DataType>
void LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::forward() {

  this->report_forward_last_transfer.reset();

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
              continue; // in the padding region, so we're adding 0
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

  this->report_forward_last_transfer.end();
  this->report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Implements LRN in the backward direction. (Note: we only support ACROSS
 * CHANNEL normalization.)
 * This is also implemented very differently from Caffe, but it should still
 * produce the same result.
 **/
template <typename DataType>
void LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::backward() {

  this->report_backward_updateweight_last_transfer.reset();

  const DataType alpha_over_size = alpha / local_size;

  for (size_t o_b = 0; o_b < iB; ++o_b) {
    for (int o_d = 0; o_d < iD; ++o_d) {
      for (size_t o_c = 0; o_c < iC; ++o_c) {
        for (size_t o_r = 0; o_r < iR; ++o_r) {
          const DataType output_grad = *p_output_layer->p_gradient_cube->logical_get(o_r, o_c, o_d, o_b);
          const DataType denom_no_exponent = *denoms->logical_get(o_r, o_c, o_d, o_b);
          const DataType denom = pow(denom_no_exponent, beta);
          const DataType input_data = *p_input_layer->p_data_cube->logical_get(o_r, o_c, o_d, o_b);
          const DataType input_grad = (denom - 2*pow(input_data,2) * beta * alpha_over_size * pow(denom_no_exponent, beta - 1)) / pow(denom, 2);
          *p_input_layer->p_gradient_cube->logical_get(o_r, o_c, o_d, o_b) = input_grad * output_grad;
        }
      }
    }
  }

  this->report_backward_updateweight_last_transfer.end();
  this->report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType>
LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::~LRNBridge() {
  delete denoms;
}

#endif
