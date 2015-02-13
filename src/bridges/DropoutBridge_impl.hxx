//
//  DropoutBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/31/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_DropoutBridge_impl_hxx
#define moka_DropoutBridge_impl_hxx

template <typename DataType>
DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::DropoutBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_p_input_layer, _p_output_layer, _layer_param),
 dropout_ratio(layer_param->dropout_param().dropout_ratio()) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(oR == iR); assert(oC == iC);
  assert(oB == iB); assert(oD == iD);
  assert(dropout_ratio > 0.);
  assert(dropout_ratio < 1.);
#endif

  scale = 1. / (1. - dropout_ratio);
  mask_cube = new LogicalCube<unsigned int, Layout_CRDB>(iR, iC, iD, iB);
  Util::bernoulli_initialize(mask_cube->p_data, iR*iC*iD*iB, 1. - dropout_ratio);

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Implements Dropout in the forward direction.
 **/
template <typename DataType>
void DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::forward() {
  report_forward_last_transfer.reset();

  const size_t num_elements = p_input_layer->p_data_cube->n_elements;
#ifdef _DO_ASSERT
  assert(num_elements == mask_cube->n_elements);
#endif
  const unsigned int * const mask = mask_cube->p_data;
  const DataType * const input_data = p_input_layer->p_data_cube->p_data;
  DataType * const output_data = p_output_layer->p_data_cube->p_data;

  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = input_data[i] * mask[i] * scale;
  }

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Implements Dropout in the backward direction.
 **/
template <typename DataType>
void DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::backward() {
  report_backward_updateweight_last_transfer.reset();

  const size_t num_elements = p_input_layer->p_data_cube->n_elements;
#ifdef _DO_ASSERT
  assert(num_elements == mask_cube->n_elements);
#endif
  const unsigned int * const mask = mask_cube->p_data;

  DataType* const input_gradient = p_input_layer->p_gradient_cube->p_data;
  const DataType* const output_gradient = p_output_layer->p_gradient_cube->p_data;

  for (size_t i = 0; i < num_elements; ++i) {
    input_gradient[i] = output_gradient[i] * mask[i] * scale;
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType>
DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::~DropoutBridge() {
  delete mask_cube;
}

#endif
