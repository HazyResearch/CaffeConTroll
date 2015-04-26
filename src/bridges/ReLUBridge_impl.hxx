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
  // Copy input to device memory
  AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(input_d_cube,
      p_input_layer->p_data_cube);
  // If DriverClass == CPUDriver, we also need to update the p_data pointer of output_d_cube to point to
  // p_output_layer->p_data_cube->p_data
  if (std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(
        output_d_cube, p_output_layer->p_data_cube
        );
  }

  report_forward_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  DeviceMemoryPointer * input = input_d_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);

  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer(NULL, 0);
  DeviceMemoryPointer * arg2 = p_driver->get_device_pointer(NULL, 0);

  p_driver->template parallel_map<_f_src_to_dst_relu_forward,
    _f_relu_forward>(input, output, sizeof(DataType), arg1, arg2);
  ////////////////////////////////////////////////////////////////////////////////

  // If DriverClass == GPUDriver (or DriverClass != CPUDriver), we copy output to host memory here
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_device_to_local(
        p_output_layer->p_data_cube, output_d_cube
        );
  }

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}

/**
 * Implements ReLU in the backward direction. (Note: we don't support
 * a negative slope parameter yet, like Caffe.)
 **/
template <typename DataType, typename DriverClass>
void ReLUBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {
  // Copy output grad to device memory
  AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(output_g_cube,
      p_output_layer->p_gradient_cube);
  // If DriverClass == CPUDriver, we also need to update the p_data pointer of input_g_cube to point to
  // p_input_layer->p_gradient_cube->p_data
  if (std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(
        input_g_cube, p_input_layer->p_gradient_cube
        );
  }

  report_backward_updateweight_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  DeviceMemoryPointer * input = input_g_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_g_cube->get_device_pointer(p_driver);

  _relu_backward_arg_helper _arg;
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(input_d_cube,
        p_input_layer->p_data_cube);
    _arg.input_data = (char *) input_d_cube->get_p_data();
  } else {
    _arg.input_data = (char *) p_input_layer->p_data_cube->get_p_data();
  }


  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer(NULL, 0);
  DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_relu_backward_arg_helper));

  p_driver->template parallel_map<_f_src_to_dst_relu_backward,
    _f_relu_backward>(input, output, sizeof(DataType), arg1, arg2);
  ////////////////////////////////////////////////////////////////////////////////

  // If DriverClass == GPUDriver (or DriverClass != CPUDriver), we copy input grad to host memory here
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_device_to_local(
        p_input_layer->p_gradient_cube, input_g_cube
        );
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

#endif
