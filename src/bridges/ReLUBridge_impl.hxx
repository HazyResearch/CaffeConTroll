//
//  ReLUBridge_impl.hxx
//
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

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  input_d_cube ->set_p_data(p_input_layer ->p_data_cube->get_p_data());
  output_d_cube->set_p_data(p_output_layer->p_data_cube->get_p_data());

  report_forward_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  PROFILE_ONLY(p_driver->device_sync(); Timer t;)
  
  DeviceMemoryPointer * input = input_d_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);

  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer(NULL, 0);
  DeviceMemoryPointer * arg2 = p_driver->get_device_pointer(NULL, 0);

  p_driver->template parallel_map<_f_src_to_dst_relu_forward,   // SHADJIS TODO: make faster CPU
    _f_relu_forward>(input, output, sizeof(DataType), arg1, arg2);
    
  PROFILE_ONLY(p_driver->device_sync(); float seconds = t.elapsed(); std::cout << "  Fw ReLU        " << seconds << "\n";)
  ////////////////////////////////////////////////////////////////////////////////

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}

/**
 * Implements ReLU in the backward direction. (Note: we don't support
 * a negative slope parameter yet, like Caffe.)
 **/
template <typename DataType, typename DriverClass>
void ReLUBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  output_g_cube->set_p_data(p_output_layer->p_gradient_cube->get_p_data());
  input_g_cube ->set_p_data(p_input_layer ->p_gradient_cube->get_p_data());

  report_backward_updateweight_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  PROFILE_ONLY(p_driver->device_sync(); Timer t;)
  
  DeviceMemoryPointer * input = input_g_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_g_cube->get_device_pointer(p_driver);

  _relu_backward_arg_helper _arg;
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    // This was already copied to device in the fw pass
    _arg.input_data = (char *) input_d_cube->get_p_data();
  } else {
    _arg.input_data = (char *) p_input_layer->p_data_cube->get_p_data();
  }


  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer(NULL, 0);
  DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_relu_backward_arg_helper));

  p_driver->template parallel_map<_f_src_to_dst_relu_backward,   // SHADJIS TODO: make faster CPU
    _f_relu_backward>(input, output, sizeof(DataType), arg1, arg2);
    
  PROFILE_ONLY(p_driver->device_sync(); float seconds = t.elapsed(); std::cout << "  Bw ReLU        " << seconds << "\n";)
  ////////////////////////////////////////////////////////////////////////////////

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

#endif
