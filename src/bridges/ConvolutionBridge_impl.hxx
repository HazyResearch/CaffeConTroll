//
//  ConvolutionBridge_impl.hxx
//  moka
//
//  Created by Ce Zhang on 1/13/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

//#include "../sched/DeviceDriver_GPU.h"

#ifndef moka_ConvolutionBridge_impl_hxx
#define moka_ConvolutionBridge_impl_hxx

// Constructor for convolution layer
template <typename DataType, NonLinearFunction FUNC, typename DriverClass>
ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
ConvolutionBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
  const cnn::LayerParameter * const _layer_param, const cnn::SolverParameter * const _solver_param,
  DriverClass * const _p_driver) : AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>(_p_input_layer,
    _p_output_layer, _layer_param, _solver_param, _p_driver), K(layer_param->convolution_param().kernel_size()),
  num_output_features(_p_output_layer->dD),
  // We are missing the abstraction of Logical Plan -- that is
  // why we cannot use layer_param here when there is grouping.
  // layer_param is the user input, not the Logical Plan
  stride(layer_param->convolution_param().stride()),
  padding(layer_param->convolution_param().pad()),
  bias_term(layer_param->convolution_param().bias_term()),
  weight_filler(layer_param->convolution_param().weight_filler()),
  bias_filler(layer_param->convolution_param().bias_filler()) {

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
    // this should be POINT to device

  p_model_cube_shadow = new LogicalCubeType(K, K, iD, num_output_features, p_driver);
    // this should be allocated on device

  initialize_logical_cube(p_model_cube_shadow, weight_filler);

  p_model_gradient_cube = new LogicalCubeType(K, K, iD, num_output_features, p_driver);
    // this should be allocated on device
  p_driver->sconstant_initialize(p_model_gradient_cube->get_device_pointer(p_driver), DataType(0.));

  if (bias_term) {
    p_bias_cube = new LogicalCubeType(1, 1, num_output_features, 1, p_driver);
        // this should be allocated to device
    initialize_logical_cube(p_bias_cube, bias_filler);

    p_bias_gradient_cube = new LogicalCubeType(1, 1, num_output_features, 1, p_driver);
      // this should be allocated on device
  }

  // First, allocate the space we need for lowering
  // Following code is very messy without the Matrix interface -- TODO
  p_forward_lowered_data = new LogicalCube<DataType, Layout_CRDB>(K*K*iD, oR*oC*iB,
      1, 1, p_driver);
      // this should be allocated on device
  DeviceMemoryPointer * p_lowered_result = p_forward_lowered_data->get_device_pointer(p_driver);
  p_driver->sconstant_initialize(p_lowered_result, (DataType) 0.0);

  LogicalCube<DataType, Layout_CRDB> lowered_forward_model(p_model_cube_shadow->get_p_data(), num_output_features,
      K*K*iD, 1, 1);
    // this should be POINT to device

  LogicalCube<DataType, Layout_CRDB> lowered_forward_output(p_output_layer->p_data_cube->get_p_data(),
      num_output_features, oR*oC*iB, 1, 1);
    // this should be POINT to device

  p_forward_lower_connector = new Connector<DataType, Layout_CRDB, DataType, Layout_CRDB,
                            LOWERING_TYPE1, DriverClass>(p_input_layer->p_data_cube, p_forward_lowered_data, K,
                                padding, stride, p_driver);

  p_forward_gemm_kernel = new Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB,
                        Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_NOTRANS>(&lowered_forward_model,
                            p_forward_lowered_data, &lowered_forward_output, p_driver);

  p_forward_applyfunc_scanner = new Scanner<DataType, Layout_CRDB, FUNC>(p_output_layer->p_data_cube,
                                  p_driver);

  // second, allocate the space we need for backward
  // (only if we're applying a non-linear function
  // after the convolution)
  if (FUNC != FUNC_NOFUNC) {
    p_backward_outputgrad = new LogicalCube<DataType, Layout_CRDB>(oR, oC, oD, oB, p_driver);
      // this should be allocated on device
  }

  p_backward_inputgrad = new LogicalCube<DataType, Layout_CRDB>(K*K*iD, oR*oC*iB, 1, 1, p_driver);
      // this should be allocated on device

  // TODO: figure out a better way to support other functions besides tanh
  if (FUNC != FUNC_NOFUNC) {
    p_backward_element_mul_kernel = new Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType,
                                  Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU,
                                  KernelConfig_TANHGRAD_ON_INPUT1>(p_output_layer->p_data_cube,
                                      p_output_layer->p_gradient_cube, p_backward_outputgrad, p_driver);
  }

  // TODO: this constructor doesn't make any sense -- we're passing in different arguments later, in backward()
  p_backward_gemm_updateweight_kernel = new Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType,
                                      Layout_CRDB, Kernel_GEMM_OpenBlas,
                                      KernelConfig_GEMM_NOTRANS_TRANS>(&lowered_forward_output,
                                          p_forward_lowered_data, &lowered_forward_model, p_driver);

  p_backward_gemm_updategrad_kernel = new Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB,
                                    DataType_SFFloat, Layout_CRDB, Kernel_GEMM_OpenBlas,
                                    KernelConfig_GEMM_TRANS_NOTRANS>(&lowered_forward_model,
                                        &lowered_forward_output, p_backward_inputgrad, p_driver);

  report_forward_constructor.end(0, 0, 0);
}

// Intiailize a Logical Cube using a FillerParameter. This is only called if layer_param is
// non-NULL.
template <typename DataType, NonLinearFunction FUNC, typename DriverClass>
void ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
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
 * (3) oData -----non-linear func (if any)-----> oData
 *
 **/
template <typename DataType, NonLinearFunction FUNC, typename DriverClass>
void ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
forward() {
  Util::set_num_threads(run_with_n_threads);

  // Copy input to Device. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal(p_input_layer->p_data_cube->get_p_data(),
    input_d_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost = p_driver->get_device_pointer(input_d_cube->get_p_data(),
    input_d_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(phost, &plocal);

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

  // (2) call GEMM kernel
  p_forward_gemm_kernel->compute(&lowered_model, p_forward_lowered_data, &lowered_output);

  // (3) apply non-linear functions
  if (FUNC != FUNC_NOFUNC) {
     p_forward_applyfunc_scanner->apply(&lowered_output);
  }

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

  // TODO Refactor the following code into another module
  // This code is here mainly to speed-up the refactoring
  // to bring CONV logical
  //
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
  ////////////////////////////////////////////////////////////////////////////////

  // Copy output to Host. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal2(p_output_layer->p_data_cube->get_p_data(),
    output_d_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost2 = p_driver->get_device_pointer(output_d_cube->get_p_data(),
    output_d_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(&plocal2, phost2);

  report_forward_last_transfer.end();
  report_forward_last_transfer.aggregate_onlystat(p_forward_gemm_kernel->report_last_lowering);
  report_forward_last_transfer.aggregate_onlystat(p_forward_lower_connector->report_last_lowering);

  if (FUNC != FUNC_NOFUNC) {
    report_forward_last_transfer.aggregate_onlystat(p_forward_applyfunc_scanner->report_last_apply);
  }

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
template <typename DataType, NonLinearFunction FUNC, typename DriverClass>
void ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
backward() {
  Util::set_num_threads(run_with_n_threads);

  // Copy output grad to Device. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal(p_output_layer->p_gradient_cube->get_p_data(),
      output_g_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost = p_driver->get_device_pointer(output_g_cube->get_p_data(),
      output_g_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(phost, &plocal);

  report_backward_updateweight_last_transfer.reset();

  // (1) calculate the gradient of output and store in the buffer
  if (FUNC != FUNC_NOFUNC) {
    p_backward_element_mul_kernel->compute(p_output_layer->p_data_cube, output_g_cube,
        p_backward_outputgrad);
  } else {
    p_backward_outputgrad = output_g_cube;
  }

  // (2) calculate the GEMM between the gradient of output and old kernel to calc the update on grad
  // Note: lowered_model is storing p_model_cube_history, not p_model_cube. We need this for the momentum
  // update.
  LogicalCube<DataType, Layout_CRDB> lowered_model(p_model_cube->get_p_data(), num_output_features, K*K*iD, 1, 1);
  LogicalCube<DataType, Layout_CRDB> lowered_model_grad(p_model_gradient_cube->get_p_data(), num_output_features, K*K*iD, 1, 1);
  LogicalCube<DataType, Layout_CRDB> lowered_outputgrad(p_backward_outputgrad->get_p_data(), num_output_features, oR*oC*iB, 1, 1);

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
  p_forward_lower_connector->remap_output(*p_backward_outputgrad, oB, num_output_features, oR*oC);

  if (needs_to_calc_backward_grad) {
    //    - 2.1 GEMM between the gradient of output and old kernel
    p_backward_gemm_updategrad_kernel->compute(&lowered_model, &lowered_outputgrad, p_backward_inputgrad);
    //    - 2.2 undo the lowering (i.e., sum together all grad corresponding to the same unlowered position)
    p_forward_lower_connector->inverse_lower_cube(p_backward_inputgrad, input_g_cube);
  }

  // (4) calculate the GEMM between the gradient of output and lowered data to calc the update on kernel
  p_backward_gemm_updateweight_kernel->alpha = 1.0;
  p_backward_gemm_updateweight_kernel->beta = 0.0;
  p_backward_gemm_updateweight_kernel->compute(&lowered_outputgrad, p_forward_lowered_data, &lowered_model_grad);

  report_backward_updateweight_last_transfer.end();

  if (FUNC != FUNC_NOFUNC) {
    report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_element_mul_kernel->report_last_lowering);
  }

  // Copy input grad to Host. This should be refactor'ed out into the
  // scheduler.
  DeviceMemoryPointer_Local_RAM plocal2(p_input_layer->p_gradient_cube->get_p_data(),
      input_g_cube->n_elements*sizeof(DataType));
  DeviceMemoryPointer * phost2 = p_driver->get_device_pointer(input_g_cube->get_p_data(),
      input_g_cube->n_elements*sizeof(DataType));
  p_driver->memcpy(&plocal2, phost2);

  report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_gemm_updategrad_kernel->report_last_lowering);
  report_backward_updateweight_last_transfer.aggregate_onlystat(p_forward_lower_connector->report_last_inverse_lowering);
  report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_gemm_updateweight_kernel->report_last_lowering);

  report_backward_inverse_lowering.aggregate(p_forward_lower_connector->report_last_inverse_lowering);
  report_backward_weight_kernel.aggregate(p_backward_gemm_updateweight_kernel->report_last_lowering);
  report_backward_grad_kernel.aggregate(p_backward_gemm_updategrad_kernel->report_last_lowering);
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, NonLinearFunction FUNC, typename DriverClass>
ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
~ConvolutionBridge() {
  if (FUNC != FUNC_NOFUNC) {
    delete p_backward_element_mul_kernel;
    delete p_backward_outputgrad;
  }
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
