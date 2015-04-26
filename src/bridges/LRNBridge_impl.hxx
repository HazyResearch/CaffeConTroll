//
//  LRNBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/31/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LRNBridge_impl_hxx
#define moka_LRNBridge_impl_hxx

template <typename DataType, typename DriverClass>
LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::LRNBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
    const cnn::SolverParameter * const _solver_param, DriverClass * const _p_driver)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>(_p_input_layer,
    _p_output_layer, _layer_param, _solver_param, _p_driver),
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
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    denoms_device = new LogicalCube<DataType, Layout_CRDB>(iR, iC, iD, iB, p_driver);
  } else {
    denoms_device = NULL;
  }
  report_forward_constructor.end(0, 0, 0);
}

/**
 * Implements LRN in the forward direction. (Note: we only support ACROSS
 * CHANNEL normalization.)
 * This is implemented very differently from Caffe, but it should still
 * produce the same result.
 **/
template <typename DataType, typename DriverClass>
void LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
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

  // Copy denoms to device
  
  _lrn_forward_arg_helper _arg2;
  _arg2.iR = iR;
  _arg2.iC = iC;
  _arg2.iD = iD;
  _arg2.norm_window = (int) local_size / 2;
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(denoms_device, denoms);
    _arg2.denoms = (char *) denoms_device->get_p_data();
  }
  else {
    _arg2.denoms = (char *) denoms->get_p_data();
  }

  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer(NULL, 0);
  DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg2,
      sizeof(_lrn_forward_arg_helper));

  // calculate denoms
  p_driver->template parallel_map<_f_src_to_dst_lrn_forward,
    _f_lrn_forward>(input, output, sizeof(DataType)*iR*iC*iD, arg1, arg2);

  _lrn_forward_normalize_arg_helper _arg3;
  _arg3.alpha_over_size = alpha / local_size;
  _arg3.beta = beta;
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    _arg3.denoms = (char *) denoms_device->get_p_data();
  }
  else {
    _arg3.denoms = (char *) denoms->get_p_data();
  }

  DeviceMemoryPointer * arg3 = p_driver->get_device_pointer((void*)&_arg3,
      sizeof(_lrn_forward_normalize_arg_helper));

  // then do normalization
  p_driver->template parallel_map<_f_src_to_dst_lrn_forward,
    _f_lrn_forward_normalize>(input, output, sizeof(DataType), arg1, arg3);
  ////////////////////////////////////////////////////////////////////////////////

  // If DriverClass == GPUDriver (or DriverClass != CPUDriver), we copy output to host memory here
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_device_to_local(
        p_output_layer->p_data_cube, output_d_cube
        );
  }
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_device_to_local(denoms, denoms_device);
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
template <typename DataType, typename DriverClass>
void LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {
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
  p_driver->sconstant_initialize(input, DataType(0.));

  _lrn_backward_arg_helper _arg;
  _arg.oR = oR;
  _arg.oC = oC;
  _arg.oD = oD;
  _arg.alpha_over_size = alpha / local_size;
  _arg.beta = beta;
  _arg.norm_window = (int) local_size / 2;
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(denoms_device, denoms);
    _arg.denoms = (char *) denoms_device->get_p_data();
  } else {
    _arg.denoms = (char *) denoms->get_p_data();
  }
  if (!std::is_same<DriverClass, CPUDriver>::value) {
	AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(input_d_cube,
		p_input_layer->p_data_cube);
    _arg.input_data = (char *) input_d_cube->get_p_data();
  } else {
    _arg.input_data = (char *) p_input_layer->p_data_cube->get_p_data();
  }

  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer(NULL, 0);
  DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_lrn_backward_arg_helper));

  p_driver->template parallel_map<_f_src_to_dst_lrn_backward,
    _f_lrn_backward>(input, output, sizeof(DataType)*oR*oC*oD, arg1, arg2);
  ////////////////////////////////////////////////////////////////////////////////

  // If DriverClass == GPUDriver (or DriverClass != CPUDriver), we copy input grad to host memory here
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_device_to_local(
        p_input_layer->p_gradient_cube, input_g_cube
        );
  }
  // if (!std::is_same<DriverClass, CPUDriver>::value) {
    // AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_device_to_local(denoms, denoms_device);
  // }
  // if (!std::is_same<DriverClass, CPUDriver>::value) {
	// AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_device_to_local(p_input_layer->p_data_cube, input_d_cube);
  // }
  
  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
LRNBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~LRNBridge() {
  delete denoms;
  if (denoms_device) {
	delete denoms_device;
  }
}

#endif
