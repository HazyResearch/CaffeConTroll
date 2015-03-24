//
//  ReLUBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/25/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_ReLUBridge_impl_hxx
#define moka_ReLUBridge_impl_hxx

template <typename DataType, typename DriverClass>
ReLUBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::ReLUBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
    const cnn::SolverParameter * const _solver_param, DriverClass * const _p_driver)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>(_p_input_layer, _p_output_layer,
    _layer_param, _solver_param, _p_driver) {

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
template <typename DataType, typename DriverClass>
void ReLUBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
  // Copy input to Device. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal(p_input_layer->p_data_cube->get_p_data(),
    input_d_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost = p_driver->get_device_pointer(input_d_cube->get_p_data(),
    input_d_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(phost, &plocal);

  report_forward_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  DeviceMemoryPointer * input = input_d_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);

  DeviceMemoryPointer * arg1 = NULL;
  DeviceMemoryPointer * arg2 = NULL;

  p_driver->template parallel_map<_f_src_to_dst_relu_forward,
    _f_relu_forward>(input, output, sizeof(DataType), arg1, arg2);
  ////////////////////////////////////////////////////////////////////////////////

  // Copy output to Host. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal2(p_output_layer->p_data_cube->get_p_data(),
    output_d_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost2 = p_driver->get_device_pointer(output_d_cube->get_p_data(),
    output_d_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(&plocal2, phost2);

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Implements ReLU in the backward direction. (Note: we don't support
 * a negative slope parameter yet, like Caffe.)
 **/
template <typename DataType, typename DriverClass>
void ReLUBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {
  report_backward_updateweight_last_transfer.reset();

  const size_t num_elements = p_input_layer->p_data_cube->n_elements;
  const DataType* const input_data = p_input_layer->p_data_cube->get_p_data();

  DataType* const input_gradient = p_input_layer->p_gradient_cube->get_p_data();
  const DataType* const output_gradient = p_output_layer->p_gradient_cube->get_p_data();

  for (size_t i = 0; i < num_elements; ++i) {
    input_gradient[i] = output_gradient[i] * (input_data[i] > 0);
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

#endif
