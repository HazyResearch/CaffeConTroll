//
//  MaxPoolingBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/25/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_MaxPoolingBridge_impl_hxx
#define moka_MaxPoolingBridge_impl_hxx

template <typename DataType, typename DriverClass>
MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
MaxPoolingBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
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

  pooled_height = static_cast<size_t>(ceil(static_cast<float>(
      iR - kernel_size) / stride)) + 1;
  pooled_width = static_cast<size_t>(ceil(static_cast<float>(
      iC  - kernel_size) / stride)) + 1;

#ifdef _DO_ASSERT
  assert(oR == pooled_height); assert(oC == pooled_width);
#endif

  // create Logical Cube to keep track of indices for max values
  max_index = new LogicalCube<size_t, Layout_CRDB>(pooled_height, pooled_width, iD, iB);

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Forward direction for max pooling
 **/
template <typename DataType, typename DriverClass>
void MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
  // Copy input to Device. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal(p_input_layer->p_data_cube->get_p_data(),
    input_d_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost = p_driver->get_device_pointer(input_d_cube->get_p_data(),
    input_d_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(phost, &plocal);

  report_forward_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  p_output_layer->p_data_cube->reset_cube(-FLT_MAX);

  DeviceMemoryPointer * input = input_d_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);

  _pool_forward_arg_helper _arg;
  _arg.pooled_height = pooled_height;
  _arg.pooled_width = pooled_width;
  _arg.stride = stride;
  _arg.kernel_size = kernel_size;
  _arg.iR = iR;
  _arg.iC = iC;
  _arg.oR = oR;
  _arg.oC = oC;
  _arg.max_index = (char *) max_index->get_p_data();

//  const int inc_input = input_data->R*input_data->C;
//  const int inc_output = output_data->R*output_data->C;
//  const int inc_max = max_index->R*max_index->C;

//  input_data_pdata += inc_input;
//  output_data_pdata += inc_output;
//  max_index_slice_pdata += inc_max;

  DeviceMemoryPointer * arg1 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_dropout_forward_train_arg_helper));
  DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg,
      sizeof(_dropout_forward_train_arg_helper));

  p_driver->template parallel_map<_f_src_to_dst_pool_forward,
    _f_pool_forward>(output, input, sizeof(DataType)*iR*iC, arg1, arg2);
  ////////////////////////////////////////////////////////////////////////////////

  // Copy output to Host. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal2(p_output_layer->p_data_cube->get_p_data(),
    output_d_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost2 = p_driver->get_device_pointer(output_d_cube->get_p_data(),
    output_d_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(&plocal2, phost2);

  report_forward_last_transfer.end(1.0*iB*iD*iR*iC*sizeof(DataType),
          iB*iD*pooled_height*pooled_width*(sizeof(DataType)+sizeof(size_t)), 0);
  report_forward_history.aggregate(report_forward_last_transfer);
}

/**
 * Backward direction for max pooling. (Note: we don't handle the case of max ties)
 **/
template <typename DataType, typename DriverClass>
void MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  report_backward_updateweight_last_transfer.reset();

  p_input_layer->p_gradient_cube->reset_cube();

  const LogicalCube<DataType, Layout_CRDB>* const input_grad = p_input_layer->p_gradient_cube;
  LogicalCube<DataType, Layout_CRDB>* const output_grad = p_output_layer->p_gradient_cube;
  for (size_t b_i = 0; b_i < iB; ++b_i) {
    for (size_t d_i = 0; d_i < iD; ++d_i) {
      const LogicalMatrix<DataType> output_grad_slice = output_grad->get_logical_matrix(d_i, b_i);
      LogicalMatrix<size_t> max_index_slice = max_index->get_logical_matrix(d_i, b_i);
      LogicalMatrix<DataType> input_grad_slice = input_grad->get_logical_matrix(d_i, b_i);

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          const size_t index = ph * pooled_width + pw;
          const size_t input_grad_index = max_index_slice.p_data[index];
          input_grad_slice.p_data[input_grad_index] += output_grad_slice.p_data[index];
        }
      }
    }
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~MaxPoolingBridge() {
  delete max_index;
}

#endif
