//
//  FullyConnectedBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 2/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_FullyConnectedBridge_impl_hxx
#define moka_FullyConnectedBridge_impl_hxx

// Constructor for fully connected layer
template <typename DataType, typename DriverClass>
FullyConnectedBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
FullyConnectedBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
  const cnn::LayerParameter * const _layer_param, const cnn::SolverParameter * const _solver_param,
  DriverClass * const _p_driver) : AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB,
  DriverClass>(_p_input_layer, _p_output_layer, _layer_param, _solver_param, _p_driver),
  // padding is set to 0, and stride is set to 1. iC would also work as the
  // value set to K. (We assert that they are equal in initialize.)
  K(iR), num_output_features(layer_param->inner_product_param().num_output()),
  stride(1), padding(0), bias_term(layer_param->inner_product_param().bias_term()),
  weight_filler(layer_param->inner_product_param().weight_filler()),
  bias_filler(layer_param->inner_product_param().bias_filler()) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_kernel.reset();
  report_forward_history.reset();
  report_forward_lowering.reset();
  report_backward_inverse_lowering.reset();
  report_backward_grad_kernel.reset();
  report_backward_weight_kernel.reset();

#ifdef _DO_ASSERT
  assert(oR == (iR + 2 * padding - K) / stride + 1);
  assert(oC == (iC + 2 * padding - K) / stride + 1);
  assert(iB == oB); assert(num_output_features == oD);
#endif

  p_model_cube = new LogicalCubeType(NULL, K, K, iD, num_output_features);

  p_model_cube_shadow = new LogicalCubeType(K, K, iD, num_output_features, p_driver);
  initialize_logical_cube(p_model_cube_shadow, weight_filler);

  p_model_gradient_cube = new LogicalCubeType(K, K, iD, num_output_features, p_driver);
  p_driver->sconstant_initialize(p_model_gradient_cube->get_device_pointer(p_driver), DataType(0.));

  if (bias_term) {
    p_bias_cube = new LogicalCubeType(1, 1, num_output_features, 1, p_driver);
    initialize_logical_cube(p_bias_cube, bias_filler);

    p_bias_gradient_cube = new LogicalCubeType(1, 1, num_output_features, 1, p_driver);
  }

  // First, allocate the space we need for lowering
  // Following code is very messy without the Matrix interface -- TODO
  p_forward_lowered_data = new LogicalCube<DataType, Layout_CRDB>(K*K*iD, oR*oC*iB,
      1, 1);

  LogicalCube<DataType, Layout_CRDB> lowered_forward_model(p_model_cube_shadow->get_p_data(), num_output_features,
      K*K*iD, 1, 1);

  LogicalCube<DataType, Layout_CRDB> lowered_forward_output(p_output_layer->p_data_cube->get_p_data(),
      num_output_features, oR*oC*iB, 1, 1);

  p_forward_lower_connector = new Connector<DataType, Layout_CRDB, DataType, Layout_CRDB,
                            LOWERING_TYPE1, DriverClass>(p_input_layer->p_data_cube, p_forward_lowered_data, K,
                                padding, stride, p_driver);

  p_forward_gemm_kernel = new Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB,
                        Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_TRANS_NO_DIM_FLIP, DriverClass>(&lowered_forward_model,
                            p_forward_lowered_data, &lowered_forward_output, p_driver);

  p_backward_inputgrad = new LogicalCube<DataType, Layout_CRDB>(K*K*iD, oR*oC*iB, 1, 1);

  p_backward_gemm_updateweight_kernel = new Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType,
                                      Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_NOTRANS_DIM_FLIP,
                                      DriverClass>(&lowered_forward_output, p_forward_lowered_data,
                                          &lowered_forward_model, p_driver);
  p_backward_gemm_updateweight_kernel->alpha = 1.0;
  p_backward_gemm_updateweight_kernel->beta = 0.0;

  p_backward_gemm_updategrad_kernel = new Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB,
                                    DataType_SFFloat, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_TRANS_NOTRANS,
                                    DriverClass>(&lowered_forward_model, &lowered_forward_output, p_backward_inputgrad,
                                        p_driver);

  report_forward_constructor.end(0, 0, 0);
}

// Intiailize a Logical Cube using a FillerParameter. This is only called if layer_param is
// non-NULL.
template <typename DataType, typename DriverClass>
void FullyConnectedBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
initialize_logical_cube(const LogicalCubeType * cube, const cnn::FillerParameter filler_param) {
  const string type = filler_param.type();
  DeviceMemoryPointer * data = cube->get_device_pointer(p_driver);
  if (type == "constant") {
    p_driver->sconstant_initialize(data, (DataType) filler_param.value());
  } else if (type == "xavier") {
    p_driver->sinitialize_xavier(data, (DataType) cube->B);
  } else if (type == "bernoulli") {
    p_driver->sbernoulli_initialize(data, (DataType) filler_param.value());
  } else if (type == "gaussian") {
    p_driver->sgaussian_initialize(data, (DataType) filler_param.mean(), (DataType) filler_param.std());
  } else {
    cout << "ERROR! INITIALIZATION TYPE NOT SUPPORTED!" << endl;
    assert(false);
  }
}

/**
 * This function does the following:
 *
 * First Layer {iData, iModel, iGrad}
 * Next Layer {oData, oModel, oGrad}
 *
 * Procedure:
 *
 * (1) iData -----lowering-----> LoweredData
 *
 * (2) LoweredData x iModel -----------> oData
 *
 **/
template <typename DataType, typename DriverClass>
void FullyConnectedBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
forward() {
  p_driver->set_num_threads(run_with_n_threads);

  // PROFILE_ONLY(Timer t; float seconds_elapsed = 0.;)
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
  // PROFILE_ONLY(seconds_elapsed = t.elapsed(); std::cout << "FC PROFILE Forward Device Copy: " << seconds_elapsed << " seconds." << std::endl; t.restart();)

  report_forward_last_transfer.reset();

  ////////////////////////////////////////////////////////////////////////////////
  if (p_model_cube->get_p_data() == NULL) {
    p_model_cube->set_p_data(p_model_cube_shadow->get_p_data());
  }

  // (0) cast input model and output to matrix
  // This one should be refactored with the matrix interface
  LogicalCube<DataType, Layout_CRDB> lowered_model(p_model_cube->get_p_data(), num_output_features,
      K*K*iD, 1, 1);
  LogicalCube<DataType, Layout_CRDB> lowered_output(output_d_cube->get_p_data(),
      num_output_features, oR*oC*iB, 1, 1);

  // (1) do the lowering
  p_forward_lower_connector->lower_cube(input_d_cube, p_forward_lowered_data);
  // PROFILE_ONLY(seconds_elapsed = t.elapsed(); std::cout << "FC PROFILE Forward Lowering: " << seconds_elapsed << " seconds." << std::endl; t.restart();)

  // (2) call GEMM kernel
  p_forward_gemm_kernel->compute(&lowered_model, p_forward_lowered_data, &lowered_output);
  // PROFILE_ONLY(seconds_elapsed = t.elapsed(); std::cout << "FC PROFILE Forward Kernel: " << seconds_elapsed << " seconds." << std::endl; t.restart();)

  // Right now the output we get is of the form:
  // [(b_0, d_0), (b_1, d_0), ... , (b_n, d_0)
  //
  //  (b_0, d_m), (b_1, d_m), ... , (b_n, d_m)]
  //  we need to transpose this, so that the outputs
  //  of a single batch are contiguous in memory.
  //  For now, we will call remap_output to fix this
  //  issue.
  //
  //  TODO: figure out how to properly transpose the
  //  inputs so that we get the correct output without
  //  needing to call remap
  p_forward_lower_connector->remap_output(*output_d_cube, num_output_features, iB, oR*oC);
  // PROFILE_ONLY(seconds_elapsed = t.elapsed(); std::cout << "FC PROFILE Forward Remap: " << seconds_elapsed << " seconds." << std::endl; t.restart();)

  // add bias
  if (bias_term) {
    DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);
    DeviceMemoryPointer * bias = p_bias_cube->get_device_pointer(p_driver);

    _bias_arg_helper _arg1;
    _arg1.src_skip = oR*oC*sizeof(DataType);
    _arg1.DataTypeSize = sizeof(DataType);
    _arg1.oD = oD;

    size_t ORxOC = oR*oC;

    DeviceMemoryPointer * arg1 = p_driver->get_device_pointer((void*)&_arg1,
      sizeof(_bias_arg_helper));

    DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&ORxOC,
        sizeof(size_t));

    p_driver->template parallel_map<_f_src_to_dst_bias_forward,
      _f_bias_forward>(bias, output, _arg1.src_skip, arg1, arg2);
  }
  // PROFILE_ONLY(seconds_elapsed = t.elapsed(); std::cout << "FC PROFILE Forward Bias: " << seconds_elapsed << " seconds." << std::endl; t.restart();)
  ////////////////////////////////////////////////////////////////////////////////

  // If DriverClass == GPUDriver (or DriverClass != CPUDriver), we copy output to host memory here
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(
        p_output_layer->p_data_cube, output_d_cube
        );
  }

  report_forward_last_transfer.end();
  report_forward_last_transfer.aggregate_onlystat(p_forward_gemm_kernel->report_last_lowering);
  report_forward_last_transfer.aggregate_onlystat(p_forward_lower_connector->report_last_lowering);

  report_forward_history.aggregate(report_forward_last_transfer);
  report_forward_kernel.aggregate(p_forward_gemm_kernel->report_last_lowering);
  report_forward_lowering.aggregate(p_forward_lower_connector->report_last_lowering);
}

/**
  * This function does the following:
  *
  * First Layer {iData, iModel, iGrad}
  * Next Layer {oData, oModel, oGrad}
  *
  * Procedure:
  *
  * (1) oData element-wise-mul oGrad -------> BackPropogatedGradient
  *
  * (2) Update iGrad:
  *
  * (2.1) iModel x BackPropogatedGradient -----------> LoweredGradient_for_iData
  *
  * (2.2) LoweredGradient_for_iData ----inverse_of_lowering----> iGrad
  *
  * (3) BackPropogatedGradient x Lowered_iData * stepsize + iModel ---------> New iModel
  *
 **/
template <typename DataType, typename DriverClass>
void FullyConnectedBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
backward() {
  p_driver->set_num_threads(run_with_n_threads);

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

  // (2) calculate the GEMM between the gradient of output and old kernel to calc the update on grad
  // Note: lowered_model is storing p_model_cube_history, not p_model_cube. We need this for the momentum
  // update.
  LogicalCube<DataType, Layout_CRDB> lowered_model(p_model_cube->get_p_data(), num_output_features,
      K*K*iD, 1, 1);
  LogicalCube<DataType, Layout_CRDB> lowered_model_grad(p_model_gradient_cube->get_p_data(), num_output_features,
      K*K*iD, 1, 1);
  LogicalCube<DataType, Layout_CRDB> lowered_outputgrad(output_g_cube->get_p_data(), num_output_features,
      oR*oC*iB, 1, 1);

  // (3) update the bias term, summing over the gradients for each O and B
  if (bias_term) {
    DeviceMemoryPointer * input = input_g_cube->get_device_pointer(p_driver);
    DeviceMemoryPointer * bias = p_bias_gradient_cube->get_device_pointer(p_driver);
    p_driver->sconstant_initialize(bias, DataType(0.));

    _bias_arg_helper _arg1;
    _arg1.src_skip = oR*oC*sizeof(DataType);
    _arg1.DataTypeSize = sizeof(DataType);
    _arg1.oD = oD;

    size_t ORxOC = oR*oC;

    DeviceMemoryPointer * arg1 = p_driver->get_device_pointer((void*)&_arg1,
      sizeof(_bias_arg_helper));

    DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&ORxOC,
        sizeof(size_t));

    p_driver->template parallel_map<_f_src_to_dst_bias_backward,
      _f_bias_backward>(bias, input, _arg1.src_skip, arg1, arg2);
  }

  // Here, we again call remap_output, but we do so BEFORE calling compute and inverse_lower_cube
  p_forward_lower_connector->remap_output(*output_g_cube, oB, num_output_features, oR*oC);
  //    - 2.1 GEMM between the gradient of output and old kernel
  p_backward_gemm_updategrad_kernel->compute(&lowered_model, &lowered_outputgrad, p_backward_inputgrad);
  //    - 2.2 undo the lowering (i.e., sum together all grad corresponding to the same unlowered position)
  p_forward_lower_connector->inverse_lower_cube(p_backward_inputgrad, input_g_cube);
  // (4) calculate the GEMM between the gradient of output and lowered data to calc the update on kernel
  p_backward_gemm_updateweight_kernel->alpha = 1.0;
  p_backward_gemm_updateweight_kernel->beta = 0.0;

  p_backward_gemm_updateweight_kernel->compute(&lowered_outputgrad, p_forward_lowered_data, &lowered_model_grad);

  report_backward_updateweight_last_transfer.end();

  // If DriverClass == GPUDriver (or DriverClass != CPUDriver), we copy input grad to host memory here
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_local_to_device(
        p_input_layer->p_gradient_cube, input_g_cube
        );
  }

  report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_gemm_updategrad_kernel->report_last_lowering);
  report_backward_updateweight_last_transfer.aggregate_onlystat(p_forward_lower_connector->report_last_inverse_lowering);
  report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_gemm_updateweight_kernel->report_last_lowering);

  report_backward_inverse_lowering.aggregate(p_forward_lower_connector->report_last_inverse_lowering);
  report_backward_weight_kernel.aggregate(p_backward_gemm_updateweight_kernel->report_last_lowering);
  report_backward_grad_kernel.aggregate(p_backward_gemm_updategrad_kernel->report_last_lowering);
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
FullyConnectedBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
~FullyConnectedBridge() {
  if (bias_term) {
    delete p_bias_cube;
    delete p_bias_gradient_cube;
  }
  delete p_model_cube_shadow; delete p_model_gradient_cube; delete p_forward_lowered_data;
  delete p_backward_gemm_updategrad_kernel; delete p_backward_gemm_updateweight_kernel;
  delete p_backward_inputgrad; delete p_forward_gemm_kernel;
  delete p_forward_lower_connector;
}

#endif
