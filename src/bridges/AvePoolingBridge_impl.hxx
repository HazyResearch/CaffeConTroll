//
//  AvePoolingBridge_impl.hxx
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef _AvePoolingBridge_impl_hxx
#define _AvePoolingBridge_impl_hxx

// SHADJIS TODO: A lot of these bridges are unnecessarily complicated
// Each function has a bunch of lines which do nothing
// E.g. reports can be disabled unless -D_LAYER_PROFILE is set, etc.

template <typename DataType, typename DriverClass>
AvePoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
AvePoolingBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
    const cnn::LayerParameter * const _layer_param, const cnn::SolverParameter * const _solver_param,
    DriverClass * const _p_driver) : AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB,
  DriverClass>(_p_input_layer, _p_output_layer, _layer_param, _solver_param, _p_driver) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();

  kernel_size = layer_param->pooling_param().kernel_size();
  stride = layer_param->pooling_param().stride();

#ifdef _DO_ASSERT
  assert(iD == oD); assert(iB == oB);
#endif

  pooled_height = static_cast<size_t>((static_cast<float>(
      iR - kernel_size) / stride)) + 1;
  pooled_width = static_cast<size_t>((static_cast<float>(
      iC  - kernel_size) / stride)) + 1;

#ifdef _DO_ASSERT
  assert(oR == pooled_height); assert(oC == pooled_width);
#endif

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Forward direction for ave pooling
 **/
template <typename DataType, typename DriverClass>
void AvePoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  input_d_cube ->set_p_data(p_input_layer ->p_data_cube->get_p_data());
  output_d_cube->set_p_data(p_output_layer->p_data_cube->get_p_data());

  report_forward_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  PROFILE_ONLY(p_driver->device_sync(); Timer t;)
  
  DeviceMemoryPointer * input = input_d_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);
  p_driver->sconstant_initialize(output, DataType(0.)); // SHADJIS TODO: Not needed on GPU (initialized to 0 in kernel)

  _pool_forward_arg_helper _arg;
  _arg.stride = stride;
  _arg.kernel_size = kernel_size;
  _arg.iR = iR;
  _arg.iC = iC;
  _arg.oR = oR;
  _arg.oC = oC;
  _arg.D  = iD;
  _arg.B  = iB;

  p_driver->avepool_forward(output, input, _arg);

  PROFILE_ONLY(p_driver->device_sync(); float seconds = t.elapsed(); std::cout << "  Fw AvePool     " << seconds << "\n";)
  ////////////////////////////////////////////////////////////////////////////////

  report_forward_last_transfer.end(1.0*iB*iD*iR*iC*sizeof(DataType),
          iB*iD*oR*oC*(sizeof(DataType)+sizeof(size_t)), 0);
  report_forward_history.aggregate(report_forward_last_transfer);
}

/**
 * Backward direction for ave pooling.
 **/
template <typename DataType, typename DriverClass>
void AvePoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  output_g_cube->set_p_data(p_output_layer->p_gradient_cube->get_p_data());
  input_g_cube ->set_p_data(p_input_layer->p_gradient_cube ->get_p_data());

  report_backward_updateweight_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  PROFILE_ONLY(p_driver->device_sync(); Timer t;)
  
  DeviceMemoryPointer * input = input_g_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_g_cube->get_device_pointer(p_driver);
  p_driver->sconstant_initialize(input, DataType(0.)); // SHADJIS TODO: Not needed on GPU (initialized to 0 in kernel)

  _pool_backward_arg_helper _arg;
  _arg.stride = stride;
  _arg.kernel_size = kernel_size;
  _arg.iR = iR;
  _arg.iC = iC;
  _arg.oR = oR;
  _arg.oC = oC;
  _arg.D  = iD;
  _arg.B  = iB;
  p_driver->avepool_backward(output, input, _arg);
    
  PROFILE_ONLY(p_driver->device_sync(); float seconds = t.elapsed(); std::cout << "  Bw AvePool     " << seconds << "\n";)
  ////////////////////////////////////////////////////////////////////////////////

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
AvePoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~AvePoolingBridge() {
}

#endif
