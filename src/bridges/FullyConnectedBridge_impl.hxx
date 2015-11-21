//
//  FullyConnectedBridge_impl.hxx
//
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

  // Start Reporting
  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_kernel.reset();
  report_forward_history.reset();
  report_forward_lowering.reset();
  report_backward_inverse_lowering.reset();
  report_backward_grad_kernel.reset();
  report_backward_weight_kernel.reset();

#ifdef _DO_ASSERT
  assert(oR == 1);
  assert(oC == 1);
  assert(iB == oB);
  assert(oD == num_output_features);
#endif

  // ===========================================================================
  // Cubes which are used outside this bridge
  //
  // First, create important cubes which are accessed by parallized bridge, 
  // i.e. their lifetime extends outside the forward/backward calls in this file.
  // ===========================================================================
  
  // ---------------------------------------------------------------------------
  // Model cube -- this is important because pbridge sets this model cube
  // ---------------------------------------------------------------------------
  
  // SHADJIS TODO: For legacy reasons we have two model cubes, one allocated
  // on the device (named "shadow" for no good reason and which is actually important)
  // and one which points to the shadow (which can be removed).
  
  // Allocated on device (important, should remove "shadow" from name)
  p_model_cube_shadow = new LogicalCubeType(K, K, iD, oD, p_driver); 
  initialize_logical_cube(p_model_cube_shadow, weight_filler);

  // Allocated nowhere, points to shadow (useless, delete this)
  p_model_cube = new LogicalCubeType(NULL, K, K, iD, oD);

  // ---------------------------------------------------------------------------
  // Model gradient cube -- this is important because pbridge reads this at the end
  // ---------------------------------------------------------------------------
  
  // Allocated on the device
  p_model_gradient_cube = new LogicalCubeType(K, K, iD, oD, p_driver);
  // SHADJIS TODO: This initialization is no longer needed since I re-initialize each backward
  // pass (necessary since we sum the gradients from each image consecutively, i.e. not part of GEMM)
  p_driver->sconstant_initialize(p_model_gradient_cube->get_device_pointer(p_driver), DataType(0.));

  // Repeat the above 2 things for the bias too
  if (bias_term) {
    // This is allocated on the device
    p_bias_cube = new LogicalCubeType(1, 1, oD, 1, p_driver);
    initialize_logical_cube(p_bias_cube, bias_filler);
    // This is allocated on the device
    p_bias_gradient_cube = new LogicalCubeType(1, 1, oD, 1, p_driver);
  }


  // ===========================================================================
  // Cubes which are used inside this bridge only
  //
  // These are never read outside this bridge but are cubes allocated on the
  // device and used repeatedly throughout the bridge. For this reason we only
  // allocate them once on the device, and re-use them.
  // ===========================================================================
  
  // ---------------------------------------------------------------------------
  // Extra cubes
  // ---------------------------------------------------------------------------
  // We also need to make a vector of ones, of size oR*oC*iB = iB
  // We don't need to use a cube for this but it's easier
  // Allocated on the device
  ones_bias_vector = new LogicalCube<DataType, Layout_CRDB>(iB, 1, 1, 1, p_driver);
  p_driver->sconstant_initialize(ones_bias_vector->get_device_pointer(p_driver), (DataType) 1.);
  
  // ---------------------------------------------------------------------------
  // Additional Cubes
  // ---------------------------------------------------------------------------
  // They are not shown here, but there are 4 more additional cubes allocated on
  // the device which are used throughout this bridge. They are allocated in the
  // constructor of abstractbridge since they are common to all cubes
  // These are:
  //  - input_d_cube
  //  - input_g_cube
  //  - output_d_cube
  //  - output_g_cube
  
  
  // ---------------------------------------------------------------------------
  // Kernels
  // ---------------------------------------------------------------------------
  // SHADJIS TODO:
  // Originally here we made 3 kernels, one per GEMM. These are useful as they
  // take a cube, store its size, and then provide an easy GEMM interface for
  // that size. Need to abstract current GEMM calls in a similar way, since now
  // they require explicitly providing the input sizes, even though the cubes 
  // we are doing GEMM on already contain those sizes so it is redundant.
  // I.e. just given cube inputs (containing data + size) and transpose flags
  // determine each dimension, or even simpler, abstract away transposes too by
  // create 3 functions, one for each GEMM in the fw/bw pass.
  // Also, currently we sometimes abstract the call to BLAS through another 
  // driver call, e.g. p_driver->backward_bias() and backward_bias_fc which call 
  // GEMV internally, but other times we abstract the GEMM directly.
  // Should make interface more clear.
  model_R = oD;
  model_C = K*K*iD;
  data_R  = oD;
  data_C  = iB;

  // Finish reporting
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
    std::cout << "ERROR! INITIALIZATION TYPE NOT SUPPORTED" << std::endl;
    assert(false);
  }
}

/**
 * This function does the following:
 *
 * First Layer {iData, iModel, iGrad}
 * Next Layer {oData, oModel, oGrad}
 *
 * Procedure: Copy to device, GEMM, +bias, copy to host
 *
 **/
template <typename DataType, typename DriverClass>
void FullyConnectedBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
forward() {

  // Begin forwards step
  
  PROFILE_ONLY(p_driver->device_sync(); std::cout << "FC Forward\n"; float seconds = 0.; Timer t;)
  
  // Initialize the thread 
  p_driver->init_thread(); // Only for GPU
  p_driver->set_num_threads(run_with_n_threads); // Only for CPU
  
  // Get model cube
  if (p_model_cube->get_p_data() == NULL) {
    p_model_cube->set_p_data(p_model_cube_shadow->get_p_data());
  }

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  // SHADJIS TODO: Eventually no need for input_d_cube/etc. (the 4 cubes defined in AbstractBridge),
  // since will always be same pointer as p_input_layer->p_data_cube, etc.
  input_d_cube->set_p_data( p_input_layer->p_data_cube->get_p_data());
  output_d_cube->set_p_data(p_output_layer->p_data_cube->get_p_data());

  // Start Reporting
  // This is done after the copy since that will be refactored out of the bridge.
  report_forward_last_transfer.reset();

  // Do the GEMM. No lowering is needed for FC.
  // SHADJIS TODO: Note here we're not using kernel, just doing the GEMM.
  // In the future can abstract this with a function call or kernel object
  // which takes only cubes (including sizes) and transpose information,
  // like the one used in conv now.
  p_driver->sgemm_new(CblasNoTrans, CblasTrans, data_C, model_R, model_C, (float)1.,
      input_d_cube->get_p_data(), p_model_cube->get_p_data(), (float)0., output_d_cube->get_p_data());

  PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "    GEMM:        " << seconds << "\n"; t.restart(); )
  
  // Add bias
  if (bias_term) {
    if (std::is_same<DriverClass, CPUDriver>::value) {
      // SHADJIS TODO: Also use this GEMM for fw conv bias
      // SHADJIS TODO: We are doing this with a GEMM although there is no
      // reduction so SAXPY / parallel blocked loop could be faster (but time is very small)
      p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, data_C, model_R, 1, (float)1.,
          ones_bias_vector->get_p_data(), p_bias_cube->get_p_data(), (float)1., output_d_cube->get_p_data());
    } else {
      DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);
      DeviceMemoryPointer * bias = p_bias_cube->get_device_pointer(p_driver);
      p_driver->forward_bias(bias, output, /* oR*oC = */ 1, oD, iB);
    }
  }

  PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "    Bias:        " << seconds << "\n"; t.restart(); )

  // Finish reporting. 
  report_forward_last_transfer.end();
  
  // SHADJIS TODO: Similarly, refactor this call to destroy cuBLAS out of the bridge
  p_driver->destroy_thread();
   
  // Aggregate reports
  report_forward_history.aggregate(report_forward_last_transfer);
}

/**
  * This function does the following:
  *
  * First Layer {iData, iModel, iGrad}
  * Next Layer {oData, oModel, oGrad}
  *
  * Procedure: (SHADJIS TODO: Update comment, no lowering anymore)
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

  // Begin backwards step
  
  PROFILE_ONLY(p_driver->device_sync(); std::cout << "FC backward\n"; float seconds = 0.; Timer t;)
  
  // Initialize the thread 
  p_driver->init_thread(); // Only for GPU
  p_driver->set_num_threads(run_with_n_threads); // Only for CPU
  
  // Get model cube
  DeviceMemoryPointer * model_gradient_pointer = p_model_gradient_cube->get_device_pointer(p_driver);
  p_driver->sconstant_initialize(model_gradient_pointer, DataType(0.));

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  // SHADJIS TODO: Eventually no need for input_d_cube/etc. (the 4 cubes defined in AbstractBridge),
  // since will always be same pointer as p_input_layer->p_data_cube, etc.
  output_g_cube->set_p_data(p_output_layer->p_gradient_cube->get_p_data());
  input_g_cube->set_p_data(p_input_layer->p_gradient_cube->get_p_data());
  // SHADJIS TODO: Since this was done in FW pass and never freed from fw this currently is not needed.
  // If we do need it again, refactor the copy out to pbridge like the others were.
  //AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_host_to_device(input_d_cube,
  //    p_input_layer->p_data_cube);

  // Start Reporting
  // SHADJIS TODO: This is done after the copy since that will be refactored out of the bridge.
  report_backward_updateweight_last_transfer.reset();

  // Update the bias term, summing over the gradients for each O and B
  if (bias_term) {
    DeviceMemoryPointer * output = output_g_cube->get_device_pointer(p_driver);
    DeviceMemoryPointer * bias = p_bias_gradient_cube->get_device_pointer(p_driver);
    // Note: Because the FC bridge has oR*oC = 1x1, we don't need to call the
    // normal p_driver->backward_bias() which does 1 GEMV for each batch, and
    // sums all the GEMV results (i.e. beta=1). Instead, we can calculate as
    // a single GEMV, by transposing to place the batches next to each other
    // in memory.
    // SHADJIS TODO: Profile to see if this is faster
    // Old call: (ones_bias_vector can also be a factor of B smaller)
    //p_driver->backward_bias(bias, output, /* oR*oC = */ 1, oD, iB, ones_bias_vector->get_p_data());
    // SHADJIS TODO: No need to initialize bias anymore (since all done in 1 GEMV)
    p_driver->backward_bias_fc(bias, output, oD, iB, ones_bias_vector->get_p_data());
  }
  
  PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "    Bias:        " << seconds << "\n"; t.restart(); )

  // Calculate the GEMM between the gradient of output and old kernel to calc the update on grad
  // SHADJIS TODO: Add a check to see if fc is 1st layer and does not need backwards data
  p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, data_C, model_C, model_R, (float)1.,
      output_g_cube->get_p_data(), p_model_cube->get_p_data(), (float)0., input_g_cube->get_p_data());
  
  PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "    GEMM Data:   " << seconds << "\n"; t.restart(); )
  
  // Calculate the GEMM between the gradient of output and lowered data to calc the update on kernel
  p_driver->sgemm_new(CblasTrans, CblasNoTrans, data_R, model_C, data_C, (float)1.,
      output_g_cube->get_p_data(), input_d_cube->get_p_data(), (float)0., p_model_gradient_cube->get_p_data());
  
  PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "    GEMM Wghts:  " << seconds << "\n"; t.restart(); )
  
  // Finish reporting. 
  // SHADJIS TODO: This is done before copy back to device, since copies need 
  // to be refactored out of the bridge
  report_backward_updateweight_last_transfer.end();
  
  // SHADJIS TODO: Similarly, refactor this call to destroy cuBLAS out of the bridge
  p_driver->destroy_thread();

  // Aggregate reports
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
FullyConnectedBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
~FullyConnectedBridge() {
  if (bias_term) {
    delete p_bias_cube;
    delete p_bias_gradient_cube;
  }
  delete p_model_cube_shadow;
  delete p_model_gradient_cube;
  delete ones_bias_vector;
}

#endif
