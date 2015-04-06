//
//  SoftmaxLossBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/25/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_SoftmaxLossBridge_impl_hxx
#define moka_SoftmaxLossBridge_impl_hxx

template <typename DataType, typename DriverClass>
SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::SoftmaxLossBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, DataLabelsLogicalCubeType * const _p_data_labels, DriverClass * const _p_driver) :
  AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>(_p_input_layer, _p_output_layer, _p_driver),
  p_data_labels(_p_data_labels), ldR(p_data_labels->R), ldC(p_data_labels->C), ldD(p_data_labels->D), ldB(p_data_labels->B) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(iR==oR);  assert(iC==oC);
  assert(iB==oB);  assert(ldR==1);
  assert(ldC==1);  assert(ldD==1);
  assert(oB==ldB); //assert(oD==ldD);
#endif

  loss = DataType(0.0);

  report_forward_constructor.end(0, 0, 0);
}

/**
 * forward direction for Softmax Loss
 * TODO: Predictions need to be written to output layer
 **/
template <typename DataType, typename DriverClass>
void SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
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

  _softmax_forward_arg_helper _arg;
  _arg.loss = &loss;
  _arg.iR = iR;
  _arg.iC = iC;
  _arg.iD = iD;
  _arg.ground_truth = (char *) p_data_labels->get_p_data();

  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_softmax_forward_arg_helper));
  DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_softmax_forward_arg_helper));

  p_driver->template parallel_map<_f_src_to_dst_softmax_forward,
    _f_softmax_forward>(output, input, sizeof(DataType)*iR*iC*iD, arg1, arg2);
  ////////////////////////////////////////////////////////////////////////////////

  // If DriverClass == GPUDriver (or DriverClass != CPUDriver), we copy output to host memory here
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(
        p_output_layer->p_data_cube, output_d_cube
        );
  }

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}

/**
 * Backward propogation for Softmax Loss
 **/
template <typename DataType, typename DriverClass>
void SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {
  // If DriverClass == CPUDriver, we need to update the p_data pointer of input_g_cube to point to
  // p_input_layer->p_gradient_cube->p_data
  if (std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(
        input_g_cube, p_input_layer->p_gradient_cube
        );
  }
  // We copy the output data (in host) into the input gradient (on device)
  // (This is because output grad is empty for Softmax Loss, since it's the last layer
  // in the network.)
  DeviceMemoryPointer_Local_RAM plocal(p_output_layer->p_data_cube->get_p_data(),
      output_d_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost = p_driver->get_device_pointer(input_g_cube->get_p_data(),
      input_g_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(phost, &plocal);

  report_backward_updateweight_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  DeviceMemoryPointer * input = input_g_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_g_cube->get_device_pointer(p_driver);

  _softmax_backward_arg_helper _arg;
  _arg.iR = iR;
  _arg.iC = iC;
  _arg.iD = iD;
  _arg.iB = iB;
  _arg.ground_truth = (char *) p_data_labels->get_p_data();

  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_softmax_backward_arg_helper));
  DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_softmax_backward_arg_helper));

  p_driver->template parallel_map<_f_src_to_dst_softmax_backward,
    _f_softmax_backward>(output, input, sizeof(DataType)*iR*iC*iD, arg1, arg2);
  ////////////////////////////////////////////////////////////////////////////////

  // If DriverClass == GPUDriver (or DriverClass != CPUDriver), we copy input grad to host memory here
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(
        p_input_layer->p_gradient_cube, input_g_cube
        );
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

#endif
