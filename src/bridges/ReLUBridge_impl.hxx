//
//  ReLUBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/25/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_ReLUBridge_impl_hxx
#define moka_ReLUBridge_impl_hxx

template <typename DataType>
ReLUBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::ReLUBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_p_input_layer, _p_output_layer) {
  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(oR==iR); assert(oC==iC);
  assert(oB==iB); assert(oD==iD);
#endif

  // no-op

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Implements ReLU in the forward direction. (Note: we don't support
 * a negative slope parameter yet, like Caffe.)
 **/
template <typename DataType>
void ReLUBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::forward() {

  openblas_set_num_threads(run_with_n_threads);

  report_forward_last_transfer.reset();

  const size_t num_elements = p_input_layer->p_data_cube->n_elements;
  const DataType* const input_data = p_input_layer->p_data_cube->p_data;
  DataType* const output_data = p_output_layer->p_data_cube->p_data;

  for (size_t i = 0; i < num_elements; ++i) {
    output_data[i] = max(input_data[i], DataType(0));
  }

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Implements ReLU in the backward direction. (Note: we don't support
 * a negative slope parameter yet, like Caffe.)
 **/
template <typename DataType>
void ReLUBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::backward() {

  openblas_set_num_threads(run_with_n_threads);

  report_backward_updateweight_last_transfer.reset();

  const size_t num_elements = p_input_layer->p_data_cube->n_elements;
  const DataType* const input_data = p_input_layer->p_data_cube->p_data;

  DataType* const input_gradient = p_input_layer->p_gradient_cube->p_data;
  const DataType* const output_gradient = p_output_layer->p_gradient_cube->p_data;

  for (size_t i = 0; i < num_elements; ++i) {
    input_gradient[i] = output_gradient[i] * (input_data[i] > 0);
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

#endif
