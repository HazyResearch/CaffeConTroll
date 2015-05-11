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
  assert(oR == (iR + 2 * padding - K) / stride + 1);
  assert(oC == (iC + 2 * padding - K) / stride + 1);
  assert(iB == oB); assert(num_output_features == oD);
#endif

  // If on the GPU, cubes will be smaller (batch size 1) since we only process
  // a single batch at a time.
  size_t batch_size_device = iB;
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    batch_size_device = 1;
  }
  
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
  p_model_cube_shadow = new LogicalCubeType(K, K, iD, num_output_features, p_driver); 
  initialize_logical_cube(p_model_cube_shadow, weight_filler);

  // Allocated nowhere, points to shadow (useless, delete this)
  p_model_cube = new LogicalCubeType(NULL, K, K, iD, num_output_features);

  // ---------------------------------------------------------------------------
  // Model gradient cube -- this is important because pbridge reads this at the end
  // ---------------------------------------------------------------------------
  
  // Allocated on the device
  p_model_gradient_cube = new LogicalCubeType(K, K, iD, num_output_features, p_driver);
  // SHADJIS TODO: This initialization is no longer needed since I re-initialize each backward
  // pass (necessary since we sum the gradients from each image consecutively, i.e. not part of GEMM)
  p_driver->sconstant_initialize(p_model_gradient_cube->get_device_pointer(p_driver), DataType(0.));

  // Repeat the above 2 things for the bias too
  if (bias_term) {
    // This is allocated on the device
    p_bias_cube = new LogicalCubeType(1, 1, num_output_features, 1, p_driver);
    initialize_logical_cube(p_bias_cube, bias_filler);
    // This is allocated on the device
    p_bias_gradient_cube = new LogicalCubeType(1, 1, num_output_features, 1, p_driver);
  }


  // ===========================================================================
  // Cubes which are used inside this bridge only
  //
  // These are never read outside this bridge but are cubes allocated on the
  // device and used repeatedly throughout the bridge. For this reason we only
  // allocate them once on the device, and re-use them.
  // ===========================================================================
  
  // ---------------------------------------------------------------------------
  // Forward lowered data
  // ---------------------------------------------------------------------------
  // Allocated on the device
  // This is used both in the fw and bw pass
  // Batch Note: Currently on GPU we are redoing the fw lowering during the bw step so 
  // we only need to allocate space for a single lowered image on the device.
  p_forward_lowered_data = new LogicalCube<DataType, Layout_CRDB>(K*K*iD, oR*oC*batch_size_device,
      1, 1, p_driver);

  // ---------------------------------------------------------------------------
  // Backward data gradient before lowering
  // ---------------------------------------------------------------------------
  // Allocated on the device
  // This is used only in the bw pass but is allocated on device, so we make this
  // cube here to only allocate it once
  // Batch Note: Currently on GPU we only calculate 1 backward grad at a time so
  // we only need to allocate space for a single image on the device.
  p_backward_inputgrad = new LogicalCube<DataType, Layout_CRDB>(K*K*iD, oR*oC*batch_size_device, 1, 1, p_driver);


  // ---------------------------------------------------------------------------
  // Extra cubes
  // ---------------------------------------------------------------------------
  // On the GPU, we also need to make a vector of ones, of size oR*oC
  // We don't need to use a cube for this but it's easier
  // Allocated on the device
  if (!std::is_same<DriverClass, CPUDriver>::value) {
    ones_bias_vector = new LogicalCube<DataType, Layout_CRDB>(oR*oC, 1, 1, 1, p_driver);
    p_driver->sconstant_initialize(ones_bias_vector->get_device_pointer(p_driver), (DataType) 1.);
  }
  
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
  
  
  // ===========================================================================
  // Cubes which are used within this constructor only
  //
  // The above 2 cases handle all cubes which need to be allocated on the device.
  // The remaining cubes below are useless and only exist so we can pass some cube
  // with their size into kernel and connector constructors.
  // Put them on the stack since we never need them again.
  // Also note that these don't own their data.
  // ===========================================================================

  // A cube with the shape of the lowered foward model
  LogicalCube<DataType, Layout_CRDB> lowered_forward_model(p_model_cube_shadow->get_p_data(), num_output_features,
      K*K*iD, 1, 1);

  // A cube with the shape of the lowered forward output
  LogicalCube<DataType, Layout_CRDB> lowered_forward_output(p_output_layer->p_data_cube->get_p_data(),
      num_output_features, oR*oC*batch_size_device, 1, 1);

  // A cube with the shape of p_input_layer->p_data_cube, or input_d_cube,
  // except with batch size adjusted
  LogicalCube<DataType, Layout_CRDB> dummy_data_cube(NULL, iR, iC, iD, batch_size_device);


  // ===========================================================================
  // Connectors / Kernels
  // ===========================================================================

  p_forward_lower_connector = new Connector<DataType, Layout_CRDB, DataType, Layout_CRDB, LOWERING_TYPE1, DriverClass>
    (&dummy_data_cube, p_forward_lowered_data, K, padding, stride, p_driver);

  p_forward_gemm_kernel = new Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB,
        Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_TRANS_NO_DIM_FLIP, DriverClass>
    (&lowered_forward_model, p_forward_lowered_data, &lowered_forward_output, p_driver);

  p_backward_gemm_updateweight_kernel = new Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType,
        Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_NOTRANS_DIM_FLIP, DriverClass>
    (&lowered_forward_output, p_forward_lowered_data, &lowered_forward_model, p_driver);

  p_backward_gemm_updateweight_kernel->alpha = 1.0;
  p_backward_gemm_updateweight_kernel->beta = 0.0;
  p_backward_gemm_updategrad_kernel = new Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB,
        DataType_SFFloat, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_TRANS_NOTRANS, DriverClass>
    (&lowered_forward_model, &lowered_forward_output, p_backward_inputgrad, p_driver);

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

  // Begin forwards step
  
  // Start Reporting
  report_forward_last_transfer.reset();

  // CPU: Run in batches
  if (std::is_same<DriverClass, CPUDriver>::value) {
  
      p_driver->set_num_threads(run_with_n_threads);
      
      // Copy input to device memory
      input_d_cube->set_p_data( p_input_layer->p_data_cube->get_p_data());
          
      // If DriverClass == CPUDriver, we also need to update the p_data pointer of output_d_cube to point to
      // p_output_layer->p_data_cube->p_data
      output_d_cube->set_p_data(p_output_layer->p_data_cube->get_p_data());
      
      if (p_model_cube->get_p_data() == NULL) {
        p_model_cube->set_p_data(p_model_cube_shadow->get_p_data());
      }
      
      // Start Reporting
      // This is done after the copy since that will be refactored out of the bridge.
      PROFILE_ONLY(Timer t; Timer t_inner; float seconds;)
      report_forward_last_transfer.reset();
  
      // (0) cast input model and output to matrix
      // This one should be refactored with the matrix interface
      LogicalCube<DataType, Layout_CRDB> lowered_model(p_model_cube->get_p_data(), num_output_features,
          K*K*iD, 1, 1);
      LogicalCube<DataType, Layout_CRDB> lowered_output(output_d_cube->get_p_data(),
          num_output_features, oR*oC*iB, 1, 1);
      
      // (1) do the lowering
      // SHADJIS TODO: Pass in an argument for the lowering type (currently only type 1)
      p_forward_lower_connector->lower_cube(input_d_cube, p_forward_lowered_data);
      
      PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    Lower Cube:  " << seconds << "\n"; t_inner.restart(); )
      
      // (2) call GEMM kernel
      p_forward_gemm_kernel->compute(&lowered_model, p_forward_lowered_data, &lowered_output);
      
      PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    Remap:       " << seconds << "\n"; t_inner.restart(); )
      
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
      
      PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    GEMM:        " << seconds << "\n"; t_inner.restart(); )
      
      // add bias
      if (bias_term) {
        DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);
        DeviceMemoryPointer * bias = p_bias_cube->get_device_pointer(p_driver);
         // SHADJIS TODO: Replace this with BLAS like the GPU
        _bias_arg_helper _arg1;
        _arg1.src_skip = oR*oC*sizeof(DataType); // skip m^2, i.e. iterate for every b and for every d
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
  
      PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    Bias:        " << seconds << "\n";)
      
      // Finish reporting. 
      // SHADJIS TODO: This is done before copy back to device, since copies need 
      // to be refactored out of the bridge
      report_forward_last_transfer.end();
      PROFILE_ONLY(seconds = t.elapsed(); std::cout << "  Fw FC:         " << seconds << "\n";)
  }
  // GPU: Run 1 image at a time
  else {

      // SHADJIS TODO: Need to refactor the copies below out of the bridge
      // Similarly, refactor this call to initialize cuBLAS out of the bridge
      p_driver->init_thread();
      
      // Get model cube (do this outside loop since single model used for all images)
      if (p_model_cube->get_p_data() == NULL) {
        p_model_cube->set_p_data(p_model_cube_shadow->get_p_data());
      }
      LogicalCube<DataType, Layout_CRDB> lowered_model(p_model_cube->get_p_data(), num_output_features,
          K*K*iD, 1, 1);
      DeviceMemoryPointer * bias = p_bias_cube->get_device_pointer(p_driver);

      // Next, since we are going to lower/gemm 1 image at a time, make cubes for a single image.
      
      // Create a single-image version of input_d_cube.
      // SHADJIS TODO: This is only needed because we are still copying the entire 
      // batch to device at once. When we switch to copying 1 image at a time, this is not needed.
      LogicalCubeType input_d_cube_SINGLE_IMAGE  (NULL, // Points to 1 image of input_d_cube
        iR, iC, iD, 1, p_driver);
        
      // Create a single-image version of the lowered output
      LogicalCubeType lowered_output_SINGLE_IMAGE(NULL, // Points to 1 image of lowered output
        num_output_features, oR*oC*1, 1, 1, p_driver);

      PROFILE_ONLY(p_driver->device_sync(); std::cout << "FC Forward\n"; float seconds = 0.; Timer t;)
      
      // SHADJIS TODO: Eventually we should only copy one image at a time to the device
      // This is very easy, just
      //  1) put this copy inside the loop, and
      //  2) change the GPU specialized constructor (non-softmax one) in AbstractBridge to use batch
      //     size 1 for all 4 internal cubes
      //  3) Make a single-image (host) cube that points to 1 image at a time from p_input_layer->p_data_cube
      //  4) input_d_cube_SINGLE_IMAGE can be deleted once input_d_cube is only size 1 image
      // For now, input_d_cube is the full size so copy the entire cube.
      AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_host_to_device(input_d_cube,
          p_input_layer->p_data_cube);

      PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "  Copy local to device: " << seconds << "\n"; t.restart();)

      // Start Reporting
      // This is done after the copy since that will be refactored out of the bridge.
      report_forward_last_transfer.reset();
  
      PROFILE_ONLY(float t1 = 0.; float t2 = 0.; float t3 = 0.; Timer inner_T;)

      // Do lowering/GEMM one image at a time
      for (size_t ib = 0; ib < iB; ++ib) {
          
          // Get pointers to the right data for the lowering
          
          // SHADJIS TODO: Eventually, we will copy 1 image at a time, and then lower / gemm that
          // one image. So then, input_d_cube will have the size of a single image. For now, we are 
          // copying all images at once, so input_d_cube has the full batch size. So we needed to 
          // make a device cube of size 1 image and have it point to the right image of input_d_cube.
          // Get the right image of input_d_cube here:
          input_d_cube_SINGLE_IMAGE.set_p_data( input_d_cube->get_p_data() + ib*iR*iC*iD );
          
          // We also have to get a device cube of size 1 image to store the lowered data.
          // Recall that the forward lowering is re-used in one of the backward GEMM
          // calculations. There are 2 possibilities:
          //   1. Calculate the forward lowered data in the forward pass and also re-calculate 
          //      it in the backwards pass
          //        - No need to cuda malloc entire batch on device
          //        - But requires redoing lowering
          //        - Also, since we are redoing lowering, need to copy over data to device during
          //          backwards again (so we can re-lower it). So 
          //   2. Calculate the forward lowered data in the forward pass for each image, store
          //      all lowered images (e.g. on the device or on the host), and re-use the lowering
          //      in the backwards step.
          //        - Extra copy back in forward (copy forward lowered data to device), or no copy
          //          needed but then need to keep stored on device (which may have limited memory)
          //        - Also if transferring the data back to host at the end of fw, need to then copy
          //          lowered data back to device at beginning of bw (and lowered data is k*k times
          //          bigger than unlowered data, and on-chip bandwidth much higher)
          // Currently I am doing 1, which means that p_forward_lowered_data only ever needs to
          // store a single lowered image on the device. So, there is no need to make a SINGLE_IMAGE
          // cube like we did for input data, since p_forward_lowered_data already is size 1 image.
          
          PROFILE_ONLY(p_driver->device_sync(); inner_T.restart();)
          
          // Do the lowering
          // SHADJIS TODO: Pass in an argument for the lowering type (currently only type 1)
          p_forward_lower_connector->lower_cube(&input_d_cube_SINGLE_IMAGE, p_forward_lowered_data);
          
          PROFILE_ONLY(p_driver->device_sync(); t1 += inner_T.elapsed(); inner_T.restart();)
          
          // p_forward_lowered_data now contains a single image.
          // Next, do the GEMM and store the result in lowered_output_SINGLE_IMAGE, which stores
          // a single image of output_d_cube.
          lowered_output_SINGLE_IMAGE.set_p_data( output_d_cube->get_p_data() + ib*oR*oC*oD );

          // Call GEMM kernel
          p_forward_gemm_kernel->compute(&lowered_model, p_forward_lowered_data, &lowered_output_SINGLE_IMAGE);
          
          PROFILE_ONLY(p_driver->device_sync(); t2 += inner_T.elapsed(); inner_T.restart();)
          
          // Note: Unlike CPU, remap is not necessary here since we lower 1 image at a time only
          
          // SHADJIS TODO: Can also move this bias outside the loop, i.e. add bias to each
          // output image all at once
          if (bias_term) {
            // SHADJIS TODO: Use this line instead if moved outside
            // DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);
            // And replace the input to forward_bias with iB
            
            DeviceMemoryPointer * output = lowered_output_SINGLE_IMAGE.get_device_pointer(p_driver);
            p_driver->forward_bias(bias, output, oR*oC, oD, 1);
          }
          
          PROFILE_ONLY(p_driver->device_sync(); t3 += inner_T.elapsed(); inner_T.restart();)
      }
      
      PROFILE_ONLY(std::cout << "      FW t1 = " << t1 << "\n";)
      PROFILE_ONLY(std::cout << "      FW t2 = " << t2 << "\n";)
      PROFILE_ONLY(std::cout << "      FW t3 = " << t3 << "\n";)
      
      PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "  iB iterations:        " << seconds << "\n"; t.restart();)
  
      // Finish reporting. 
      // SHADJIS TODO: This is done before copy back to device, since copies need 
      // to be refactored out of the bridge
      report_forward_last_transfer.end();
      
      // If DriverClass == GPUDriver (or DriverClass != CPUDriver), we copy output to host memory here
      // SHADJIS TODO: In the future, if 2 consecutive bridges are on GPU, no need to
      // copy back to the host.
      AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_device_to_host(
          p_output_layer->p_data_cube, output_d_cube
        );

      PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "  Copy device to local: " << seconds << "\n";)
  
      // SHADJIS TODO: Need to refactor the copies below out of the bridge
      // Similarly, refactor this call to destroy cuBLAS out of the bridge
      p_driver->destroy_thread();
  }
   
  // Aggregate existing reports
  
  // report_forward_last_transfer.aggregate_onlystat(p_forward_gemm_kernel->report_last_lowering);
  // report_forward_last_transfer.aggregate_onlystat(p_forward_lower_connector->report_last_lowering);

  // report_forward_history.aggregate(report_forward_last_transfer);
  // report_forward_kernel.aggregate(p_forward_gemm_kernel->report_last_lowering);
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

  // Begin backwards step
  
  // CPU: Run in batches
  if (std::is_same<DriverClass, CPUDriver>::value) {
      
      p_driver->set_num_threads(run_with_n_threads);
      
      // Copy output grad to device memory
      output_g_cube->set_p_data(p_output_layer->p_gradient_cube->get_p_data());
          
      // If DriverClass == CPUDriver, we also need to update the p_data pointer of input_g_cube to point to
      // p_input_layer->p_gradient_cube->p_data
      input_g_cube->set_p_data(p_input_layer->p_gradient_cube->get_p_data());
        
      // Start Reporting
      // This is done after the copy since that will be refactored out of the bridge.
      PROFILE_ONLY(Timer t; Timer t_inner; float seconds;)
      report_backward_updateweight_last_transfer.reset();

      // Calculate the GEMM between the gradient of output and old kernel to calc the update on grad
      // Note: lowered_model is storing p_model_cube_history, not p_model_cube. We need this for the momentum update.
      LogicalCube<DataType, Layout_CRDB> lowered_model(p_model_cube->get_p_data(), num_output_features, K*K*iD, 1, 1);
      LogicalCube<DataType, Layout_CRDB> lowered_model_grad(p_model_gradient_cube->get_p_data(), num_output_features, K*K*iD, 1, 1);
      LogicalCube<DataType, Layout_CRDB> lowered_outputgrad(output_g_cube->get_p_data(), num_output_features, oR*oC*iB, 1, 1);
      
      // Update the bias term, summing over the gradients for each O and B
      if (bias_term) {
        // SHADJIS TODO: Here we used to call parallel map to do this:
        //   For each batch b
        //     For each depth d (in parallel)
        //       For each pixel p of feature map in batch b and depth d
        //         bias[d] += p
        // This can't be done with a single parallel map because of the outer batch loop.
        // On the CPU parallel_map is still used however since it is done serially. If that
        // changes, rewrite that too.
        DeviceMemoryPointer * output = output_g_cube->get_device_pointer(p_driver);
        DeviceMemoryPointer * bias = p_bias_gradient_cube->get_device_pointer(p_driver);
        p_driver->sconstant_initialize(bias, DataType(0.));
         // SHADJIS TODO: Replace this with BLAS like the GPU
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
          _f_bias_backward>(bias, output, _arg1.src_skip, arg1, arg2);
      }
      
      PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    Bias:        " << seconds << "\n"; t_inner.restart(); )
      
      // Here, we again call remap_output, but we do so BEFORE calling compute and inverse_lower_cube
      p_forward_lower_connector->remap_output(*output_g_cube, oB, num_output_features, oR*oC);
      
      PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    Remap:       " << seconds << "\n"; t_inner.restart(); )
      
      // SHADJIS TODO: Can check needs_to_calc_backward_grad here as well, e.g. maybe someone wants
      // to use CcT with just fc layers
      //    - 2.1 GEMM between the gradient of output and old kernel
      p_backward_gemm_updategrad_kernel->compute(&lowered_model, &lowered_outputgrad, p_backward_inputgrad);
      PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    GEMM Data:   " << seconds << "\n"; t_inner.restart(); )
      //    - 2.2 undo the lowering (i.e., sum together all grad corresponding to the same unlowered position)
      p_forward_lower_connector->inverse_lower_cube(p_backward_inputgrad, input_g_cube);
      PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    Inverse Lwr: " << seconds << "\n"; t_inner.restart(); )
      
      // Redo the lowering
      // Calculate the GEMM between the gradient of output and lowered data to calc the update on kernel
      p_backward_gemm_updateweight_kernel->alpha = 1.0;
      p_backward_gemm_updateweight_kernel->beta = 0.0;
      p_backward_gemm_updateweight_kernel->compute(&lowered_outputgrad, p_forward_lowered_data, &lowered_model_grad);
      
      PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    GEMM Wghts:  " << seconds << "\n"; t_inner.restart(); )
      
      // Finish reporting. 
      report_backward_updateweight_last_transfer.end();
      PROFILE_ONLY(seconds = t.elapsed(); std::cout << "  Bw FC          " << seconds << "\n";)
  }
  // GPU: Run 1 image at a time
  else {
  
      // SHADJIS TODO: Need to refactor the copies below out of the bridge
      // Similarly, refactor this call to initialize cuBLAS out of the bridge
      p_driver->init_thread();
      
      // Get model cube (do this outside loop since single model used for all images)
      LogicalCube<DataType, Layout_CRDB> lowered_model(p_model_cube->get_p_data(), num_output_features, K*K*iD, 1, 1);
      LogicalCube<DataType, Layout_CRDB> lowered_model_grad(p_model_gradient_cube->get_p_data(), num_output_features, K*K*iD, 1, 1);
      DeviceMemoryPointer * model_gradient = lowered_model_grad.get_device_pointer(p_driver);
      p_driver->sconstant_initialize(model_gradient, DataType(0.));
      DeviceMemoryPointer * bias = p_bias_gradient_cube->get_device_pointer(p_driver);
      p_driver->sconstant_initialize(bias, DataType(0.));

      // Next, since we are going to lower/gemm 1 image at a time, make cubes for a single image.

      // Create a single-image version of output_g_cube.
      // SHADJIS TODO: This is only needed because we are still copying the entire 
      // batch to device at once. When we switch to copying 1 image at a time, this is not needed,
      // as output_g_cube will already be of size 1 image.
      // For single images, the lowered output_g_cube is the same as before lowering, so use a single
      // cube for both.
      LogicalCubeType output_g_cube_SINGLE_IMAGE  (NULL, // Points to 1 image of (lowered) output_g_cube
        oD, oR*oC*1, 1, 1, p_driver);
      
      // Create a single-image version of input_g_cube.
      // We need this because we are doing lowering/GEMM on a single image at once, but at the end
      // we have to write back the entire input_g_cube to the host.
      // SHADJIS TODO: This is only needed because we are still copying the entire 
      // batch back to device at once. When we switch to copying 1 image at a time, this is not needed,
      // as input_g_cube will already be of size 1 image (and be written back each iteration).
      LogicalCubeType input_g_cube_SINGLE_IMAGE  (NULL, // Points to 1 image of (lowered) output_g_cube
        iR, iC, iD, 1, p_driver);
      
      // Create a single-image version of input_d_cube.
      // SHADJIS TODO: This is only needed because we are still copying the entire 
      // batch to device at once. When we switch to copying 1 image at a time, this is not needed.
      // SHADJIS TODO: Moreover, using input_d_cube cube is only needed in the backwards pass because 
      // one of the GEMMs requires the forward lowered data. This was previously calculated in the
      // forwards pass, but it was calculated and stored on the device only 1 image at a time, so now
      // we have to calculate it again.
      LogicalCubeType input_d_cube_SINGLE_IMAGE  (NULL, // Points to 1 image of input_d_cube
        iR, iC, iD, 1, p_driver);
      
      PROFILE_ONLY(std::cout << "FC Backward\n"; float seconds = 0.; Timer t;)
      
      // Copy output grad to device memory
      // SHADJIS TODO: Move this copy out of fc bridge. A bridge should just be passed a device memory poitner
      // or a cube and use the driver to get the next image from the cube. If the cube is already on the device,
      // i.e. all the images were copied before, then it just returns the pointer, and if it is not, it does the 
      // copy of that one image. Then the decision to copy all at once or 1 image at a time is handled be a
      // scheduler which knows the memory size of each device.
      AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_host_to_device(output_g_cube,
          p_output_layer->p_gradient_cube);

      // SHADJIS TODO: Eventually we should only copy one image at a time to the device
      // This is very easy, just
      //  1) put this copy inside the loop, and
      //  2) change the GPU specialized constructor (non-softmax one) in AbstractBridge to use batch
      //     size 1 for all 4 internal cubes
      //  3) Make a single-image (host) cube that points to 1 image at a time from p_input_layer->p_data_cube
      //  4) input_d_cube_SINGLE_IMAGE can be deleted once input_d_cube is only size 1 image
      // For now, input_d_cube is the full size so copy the entire cube.
      // Also, see comment above: only need to copy the data because we aren't copying the lowered data.
      //
      // SHADJIS TODO: Actually the data was never freed from fw so this currently is not needed
      AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_host_to_device(input_d_cube,
          p_input_layer->p_data_cube);

      PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "  Copy local to device: " << seconds << "\n"; t.restart();)

      // Start Reporting
      // SHADJIS TODO: This is done after the copy since that will be refactored out of the bridge.
      report_backward_updateweight_last_transfer.reset();

      // Calculate the GEMM between the gradient of output and old kernel to calc the update on grad
      // Note: lowered_model is storing p_model_cube_history, not p_model_cube. We need this for the momentum
      // update.
      
      PROFILE_ONLY(float t1 = 0.; float t2 = 0.; float t3 = 0.; float t4 = 0.; float t5 = 0.; Timer inner_T;)

      // Do lowering/GEMM one image at a time
      for (size_t ib = 0; ib < iB; ++ib) {
      
          // Get a single image of output_g_cube
          output_g_cube_SINGLE_IMAGE.set_p_data( output_g_cube->get_p_data() + ib*oR*oC*oD );
          
          // Also get a single image of input_g_cube
          input_g_cube_SINGLE_IMAGE.set_p_data( input_g_cube->get_p_data() + ib*iR*iC*iD );

          PROFILE_ONLY(p_driver->device_sync(); inner_T.restart();)
          
          // Update the bias term, summing over the gradients for each O and B
          // SHADJIS TODO: This is done sequentially for each image anyway, so can lift this outside loop
          if (bias_term) {
            DeviceMemoryPointer * output = output_g_cube_SINGLE_IMAGE.get_device_pointer(p_driver);
            p_driver->backward_bias(bias, output, oR*oC, oD, 1, ones_bias_vector->get_p_data());
          }
          
          PROFILE_ONLY(p_driver->device_sync(); t1 += inner_T.elapsed(); inner_T.restart();)
          
          // No need to remap since output_g_cube_SINGLE_IMAGE is only a single image
          
          // GEMM between the gradient of output and old kernel
          p_backward_gemm_updategrad_kernel->compute(&lowered_model, &output_g_cube_SINGLE_IMAGE, p_backward_inputgrad);

          PROFILE_ONLY(p_driver->device_sync(); t2 += inner_T.elapsed(); inner_T.restart();)
          
          // Undo the lowering (i.e., sum together all grad corresponding to the same unlowered position)
          p_forward_lower_connector->inverse_lower_cube(p_backward_inputgrad, &input_g_cube_SINGLE_IMAGE);
          
          PROFILE_ONLY(p_driver->device_sync(); t3 += inner_T.elapsed(); inner_T.restart();)
          
          // SHADJIS TODO: Eventually, we will copy 1 image at a time, and then lower / gemm that
          // one image. So then, input_d_cube will have the size of a single image. For now, we are 
          // copying all images at once, so input_d_cube has the full batch size. So we needed to 
          // make a device cube of size 1 image and have it point to the right image of input_d_cube.
          // Get the right image of input_d_cube here:
          input_d_cube_SINGLE_IMAGE.set_p_data( input_d_cube->get_p_data() + ib*iR*iC*iD );
          
          // Do the lowering
          // SHADJIS TODO: Pass in an argument for the lowering type (currently only type 1)
          p_forward_lower_connector->lower_cube(&input_d_cube_SINGLE_IMAGE, p_forward_lowered_data);

          PROFILE_ONLY(p_driver->device_sync(); t4 += inner_T.elapsed(); inner_T.restart();)
          
          // Calculate the GEMM between the gradient of output and lowered data to calc the update on kernel
          p_backward_gemm_updateweight_kernel->alpha = 1.0;
          p_backward_gemm_updateweight_kernel->beta = 1.0;  // 1.0 so we can accumulate gradients
          p_backward_gemm_updateweight_kernel->compute(&output_g_cube_SINGLE_IMAGE, p_forward_lowered_data, &lowered_model_grad);
          
          PROFILE_ONLY(p_driver->device_sync(); t5 += inner_T.elapsed();)
          
      }
      
      PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "  iB iterations:        " << seconds << "\n"; t.restart();)

      PROFILE_ONLY(std::cout << "      BW t1 = " << t1 << "\n";)
      PROFILE_ONLY(std::cout << "      BW t2 = " << t2 << "\n";)
      PROFILE_ONLY(std::cout << "      BW t3 = " << t3 << "\n";)
      PROFILE_ONLY(std::cout << "      BW t4 = " << t4 << "\n";)
      PROFILE_ONLY(std::cout << "      BW t5 = " << t5 << "\n";)
      
      // Finish reporting. 
      // SHADJIS TODO: This is done before copy back to device, since copies need 
      // to be refactored out of the bridge
      report_backward_updateweight_last_transfer.end();
      
      // If DriverClass == GPUDriver (or DriverClass != CPUDriver), we copy input grad to host memory here
      AbstractBridge<DataType, Layout_CRDB, DataType,Layout_CRDB, DriverClass>::copy_from_device_to_host(
          p_input_layer->p_gradient_cube, input_g_cube
        );

      PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "  Copy device to local: " << seconds << "\n";)
  
      // SHADJIS TODO: Need to refactor the copies below out of the bridge
      // Similarly, refactor this call to destroy cuBLAS out of the bridge
      p_driver->destroy_thread();
  }
  
  // Aggregate existing reports

  // report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_gemm_updategrad_kernel->report_last_lowering);
  // report_backward_updateweight_last_transfer.aggregate_onlystat(p_forward_lower_connector->report_last_inverse_lowering);
  // report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_gemm_updateweight_kernel->report_last_lowering);

  // report_backward_inverse_lowering.aggregate(p_forward_lower_connector->report_last_inverse_lowering);
  // report_backward_weight_kernel.aggregate(p_backward_gemm_updateweight_kernel->report_last_lowering);
  // report_backward_grad_kernel.aggregate(p_backward_gemm_updategrad_kernel->report_last_lowering);
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
