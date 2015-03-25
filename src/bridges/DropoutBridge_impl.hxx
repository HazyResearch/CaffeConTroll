//
//  DropoutBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/31/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_DropoutBridge_impl_hxx
#define moka_DropoutBridge_impl_hxx

template <typename DataType, typename DriverClass>
DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::DropoutBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
    const cnn::SolverParameter * const _solver_param, DriverClass * const _p_driver)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>(_p_input_layer, _p_output_layer,
    _layer_param, _solver_param, _p_driver), dropout_ratio(layer_param->dropout_param().dropout_ratio()) {

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
  Util::bernoulli_initialize(mask_cube->get_p_data(), iR*iC*iD*iB, 1. - dropout_ratio);

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Implements Dropout in the forward direction.
 **/
template <typename DataType, typename DriverClass>
void DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
  // Copy input to Device. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal(p_input_layer->p_data_cube->get_p_data(),
    input_d_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost = p_driver->get_device_pointer(input_d_cube->get_p_data(),
    input_d_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(phost, &plocal);

  report_forward_last_transfer.reset();
#ifdef _DO_ASSERT
  assert(p_input_layer->p_data_cube->n_elements == mask_cube->n_elements);
#endif

  ////////////////////////////////////////////////////////////////////////////////
  DeviceMemoryPointer * input = input_d_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);

  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer(NULL, 0);

  // in the training phase, we apply the mask
  if (DeepNetConfig::train()) {
    _dropout_forward_train_arg_helper _arg;
    _arg.mask = (char *) mask_cube->get_p_data();
    _arg.scale = scale;

    DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_dropout_forward_train_arg_helper));
    p_driver->template parallel_map<_f_src_to_dst_dropout_forward,
      _f_dropout_forward_train>(input, output, sizeof(DataType), arg1, arg2);
  // in the testing phase, we simply copy from input to output
  } else {
    DeviceMemoryPointer * arg2 = p_driver->get_device_pointer(NULL, 0);
    p_driver->template parallel_map<_f_src_to_dst_dropout_forward,
      _f_dropout_forward_test>(input, output, sizeof(DataType), arg1, arg2);
  }
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
 * Implements Dropout in the backward direction.
 **/
template <typename DataType, typename DriverClass>
void DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {
  report_backward_updateweight_last_transfer.reset();

  const size_t num_elements = p_input_layer->p_data_cube->n_elements;
#ifdef _DO_ASSERT
  assert(num_elements == mask_cube->n_elements);
#endif
  const unsigned int * const mask = mask_cube->get_p_data();

  DataType* const input_gradient = p_input_layer->p_gradient_cube->get_p_data();
  const DataType* const output_gradient = p_output_layer->p_gradient_cube->get_p_data();

  for (size_t i = 0; i < num_elements; ++i) {
    input_gradient[i] = output_gradient[i] * mask[i] * scale;
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~DropoutBridge() {
  delete mask_cube;
}

#endif
