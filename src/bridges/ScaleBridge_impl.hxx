//
//  ScaleBridge_impl.hxx
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef _ScaleBridge_impl_hxx
#define _ScaleBridge_impl_hxx

template <typename DataType, typename DriverClass>
ScaleBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::ScaleBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
    const cnn::SolverParameter * const _solver_param, DriverClass * const _p_driver)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>(_p_input_layer, _p_output_layer,
    _layer_param, _solver_param, _p_driver),
  bias_term(layer_param->scale_param().bias_term()),
  weight_filler(layer_param->scale_param().filler()),
  bias_filler(layer_param->scale_param().bias_filler()),
  p_model_gradient_cube(NULL), p_model_cube(NULL), p_model_cube_shadow(NULL), 
  p_bias_gradient_cube(NULL), p_bias_cube(NULL), ones_bias_vector(NULL), 
  sum_multiplier_(NULL), sum_result_(NULL), temp_(NULL) {

  // Start Reporting
  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(oR==iR); assert(oC==iC);
  assert(oB==iB); assert(oD==iD);
#endif

  // ===========================================================================
  // Cubes which are used outside this bridge (e.g. by pbridge)
  // ===========================================================================

  // ---------------------------------------------------------------------------
  // Model cube -- this is important because pbridge sets this model cube
  // ---------------------------------------------------------------------------
  
  // Note: We assume that this scale bridge learns a scale and bias per channel.
  // The wonderful Caffe supports other sizes, see their implementation for
  // more general scale functionality
  
  // SHADJIS TODO: For legacy reasons we have two model cubes, one allocated
  // on the device (named "shadow" for no good reason and which is actually important)
  // and one which points to the shadow (which can be removed).
  
  // Allocated on device (important, should remove "shadow" from name)
  p_model_cube_shadow = new LogicalCubeType(iD, 1, 1, 1, p_driver); 
  if (!layer_param->scale_param().has_filler()) {
    // If no filler, default to 1 (not 0, but 1 because 1 is the identity scaling)
    p_driver->sconstant_initialize(p_model_cube_shadow->get_device_pointer(p_driver), DataType(1.));
  } else {
    initialize_logical_cube(p_model_cube_shadow, weight_filler);
  }

  // Allocated nowhere, points to shadow (useless, delete this)
  p_model_cube = new LogicalCubeType(NULL, iD, 1, 1, 1);

  // ---------------------------------------------------------------------------
  // Model gradient cube -- this is important because pbridge reads this at the end
  // ---------------------------------------------------------------------------
  
  // Allocated on the device
  p_model_gradient_cube = new LogicalCubeType(iD, 1, 1, 1, p_driver);
  // SHADJIS TODO: This initialization is no longer needed since I re-initialize each backward
  // pass (necessary since we sum the gradients from each image consecutively, i.e. not part of GEMM)
  p_driver->sconstant_initialize(p_model_gradient_cube->get_device_pointer(p_driver), DataType(0.));

  // Repeat the above 2 things for the bias too
  if (bias_term) {  // SHADJIS TODO: Should assert bias term? Never did any testing without biases
    // This is allocated on the device
    p_bias_cube = new LogicalCubeType(iD, 1, 1, 1, p_driver);
    initialize_logical_cube(p_bias_cube, bias_filler);
    // This is allocated on the device
    p_bias_gradient_cube = new LogicalCubeType(iD, 1, 1, 1, p_driver);
  }


  // ===========================================================================
  // Cubes which are used inside this bridge only
  // ===========================================================================
  // We also need to make a vector of ones, of size iR*iC
  // We don't need to use a cube for this but it's easier
  // Allocated on the device
  ones_bias_vector = new LogicalCube<DataType, Layout_CRDB>(iR*iC, 1, 1, 1, p_driver);
  p_driver->sconstant_initialize(ones_bias_vector->get_device_pointer(p_driver), (DataType) 1.);

  outer_dim_ = iB;
  scale_dim_ = iD;
  inner_dim_ = iR*iC;
  // We need the input data cube for the backwards pass
  // If these are in-place, that means we need to store the input
  input_d_cube ->set_p_data(p_input_layer ->p_data_cube->get_p_data());
  output_d_cube->set_p_data(p_output_layer->p_data_cube->get_p_data());
  if (input_d_cube->get_p_data() == output_d_cube->get_p_data()) {  // in-place computation, so need a backup
    temp_ = new LogicalCube<DataType, Layout_CRDB>(iR, iC, iD, iB, p_driver);
  }
  sum_result_ = new LogicalCube<DataType, Layout_CRDB>(outer_dim_ * scale_dim_, 1, 1, 1, p_driver);
  const int sum_mult_size = std::max(outer_dim_, inner_dim_);
  sum_multiplier_ = new LogicalCube<DataType, Layout_CRDB>(sum_mult_size, 1, 1, 1, p_driver);
  p_driver->sconstant_initialize(sum_multiplier_->get_device_pointer(p_driver), DataType(1.));

  report_forward_constructor.end(0, 0, 0);
}


// Intiailize a Logical Cube using a FillerParameter. This is only called if layer_param is
// non-NULL.
template <typename DataType, typename DriverClass>
void ScaleBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::
initialize_logical_cube(const LogicalCubeType * cube, const cnn::FillerParameter filler_param) {
  const string type = filler_param.type();
  DeviceMemoryPointer * data = cube->get_device_pointer(p_driver);
  if (type == "constant") {
    p_driver->sconstant_initialize(data, (DataType) filler_param.value());
  } else if (type == "xavier") {
    p_driver->sinitialize_xavier(data, (DataType) cube->B, solver_param->random_seed());
  } else if (type == "bernoulli") {
    p_driver->sbernoulli_initialize(data, (DataType) filler_param.value(), solver_param->random_seed());
  } else if (type == "gaussian") {
    p_driver->sgaussian_initialize(data, (DataType) filler_param.mean(), (DataType) filler_param.std(), solver_param->random_seed());
  } else {
    std::cout << "ERROR! INITIALIZATION TYPE NOT SUPPORTED" << std::endl;
    assert(false);
  }
}


template <typename DataType, typename DriverClass>
void ScaleBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  input_d_cube ->set_p_data(p_input_layer ->p_data_cube->get_p_data());
  output_d_cube->set_p_data(p_output_layer->p_data_cube->get_p_data());

  report_forward_last_transfer.reset();

  // If this is a GPU bridge, init cuBLAS (does nothing on CPU)
  p_driver->init_thread();
  
  ////////////////////////////////////////////////////////////////////////////////
  PROFILE_ONLY(p_driver->device_sync(); Timer t;)
  
  // Get model cube
  if (p_model_cube->get_p_data() == NULL) {
    p_model_cube->set_p_data(p_model_cube_shadow->get_p_data());
  }

  // If computation is in-place, need to store input data into temp buffer
  if (input_d_cube->get_p_data() == output_d_cube->get_p_data()) {
    p_driver->memcpy(temp_->get_device_pointer(p_driver), input_d_cube->get_device_pointer(p_driver));
  }
  
  // Now that we are sure the input is saved and the output matches the input,
  // scale the output and then add a bias, by channel.
  
  // CPU: This is done using a number of BLAS calls
  if (std::is_same<DriverClass, CPUDriver>::value) {

    // First scale each channel (i.e. all iR*iC pixels) by that channel's weight
    
    // Could either call scale, which does a memcpy and then a scale in place, or just do the
    // memcpy first and then only scale in place. I think it should not really make a difference.
    if (input_d_cube->get_p_data() != output_d_cube->get_p_data()) {
      p_driver->memcpy(output_d_cube->get_device_pointer(p_driver), input_d_cube->get_device_pointer(p_driver));
    }
    // Now scale output in-place
    float * output = output_d_cube->get_p_data();
    for (int n = 0; n < outer_dim_; ++n) {    // outer_dim_ = iB
      for (int d = 0; d < scale_dim_; ++d) {  // scale_dim_ = iD
        const float factor = p_model_cube->get_p_data()[d];
        // we can scale in-place (no need to copy then scale) because we copied above
        p_driver->sscale_inplace(inner_dim_, factor, output);
        output += inner_dim_;
      }
    }
    // Next add a bias to the feature map at each channel
    // Note: this can be done efficiently for each image using a GEMM (on CPU)
    // Recall for each example, the output data is a 3D tensor iR x iC x iD
    // However we can view this as a 2D matrix, which is iD x (iR*iC), i.e.
    // iD rows and (iR*iC) columns, call this O. Then the GEMM does:
    //   O +=  (b1 b2 b3 ... b_iD)^T * (1 1 1 ... 1), i.e. we do a GEMM of 2
    // 1D vectors which results in the bias matrix where each row is a bias
    // term (all cols same) and the # rows is iD. So then just adding them
    // implements the bias.
    // Note: we can't do as 1 GEMM here because not contiguous in memory
    // Note: we accumulate but do not init to 0 since it adds to scaling above
    if (bias_term) {
      output = output_d_cube->get_p_data();
      for (int n = 0; n < outer_dim_; ++n) {
        p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, scale_dim_,
            inner_dim_, 1, float(1), p_bias_cube->get_p_data(),
            ones_bias_vector->get_p_data(), float(1), output);
        output += scale_dim_*inner_dim_;
      }
    }
  }
  // GPU: This can be done exactly using same GEMM calls as CPU but can also be
  // done using a single kernel call
  else {
    if (bias_term) {
      p_driver->forward_scale_and_bias(iR*iC*iD*iB, input_d_cube->get_p_data(), p_model_cube->get_p_data(),
        p_bias_cube->get_p_data(), scale_dim_, inner_dim_, output_d_cube->get_p_data());
    } else {
      p_driver->forward_scale(iR*iC*iD*iB, input_d_cube->get_p_data(), p_model_cube->get_p_data(),
        scale_dim_, inner_dim_, output_d_cube->get_p_data());
    }
  }
    
  PROFILE_ONLY(p_driver->device_sync(); float seconds = t.elapsed(); std::cout << "  Fw Scale        " << seconds << "\n";)
  ////////////////////////////////////////////////////////////////////////////////

  // If this is a GPU bridge, destroy cuBLAS (does nothing on CPU)
  p_driver->destroy_thread();

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}


template <typename DataType, typename DriverClass>
void ScaleBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  output_g_cube->set_p_data(p_output_layer->p_gradient_cube->get_p_data());
  input_g_cube ->set_p_data(p_input_layer ->p_gradient_cube->get_p_data());
  
  report_backward_updateweight_last_transfer.reset();

  // If this is a GPU bridge, init cuBLAS (does nothing on CPU)
  p_driver->init_thread();
  
  ////////////////////////////////////////////////////////////////////////////////
  PROFILE_ONLY(p_driver->device_sync(); Timer t;)
    

  // The bias has no data diff, since it was done in-place and does not
  // contribute to loss gradient wrt data, but does have a weight gradient
  // Note: we can't do as 1 GEMM here because not contiguous in memory
  if (bias_term) {
    //if (std::is_same<DriverClass, CPUDriver>::value) {
      const float* output_grad = output_g_cube->get_p_data();
      float accum = 0.;
      for (int n = 0; n < outer_dim_; ++n) {
        p_driver->sgemv(CblasNoTrans, scale_dim_, inner_dim_, float(1),
            output_grad, ones_bias_vector->get_p_data(), float(accum), p_bias_gradient_cube->get_p_data());
        output_grad += iR * iC * iD;
        accum = 1.;
      }
    //}
    // // This is equivalent, i.e. internally it does the same thing (calls gemv), so not needed
    // // Should also remove it for other bridges -- not sure why p_driver->backward_bias is ever needed
    // else {
    //  DeviceMemoryPointer * output = output_g_cube->get_device_pointer(p_driver);
    //  DeviceMemoryPointer * bias = p_bias_cube->get_device_pointer(p_driver);
    //  p_driver->backward_bias(bias, output, oR*oC, oD, iB, ones_bias_vector->get_p_data());
    //}
  }
  
  // Model gradient
  const bool in_place = (input_d_cube->get_p_data() == output_d_cube->get_p_data());
  // input_data is needed below, but it might have been overwritten if this bridge
  // is in-place, so read that stored input data from temp
  const float* input_data = (in_place ? temp_ : input_d_cube)->get_p_data();
  // Normally store the product in the input gradient cube, but if that shares the output grad cube
  // need to store it in a temp buffer instead. This is because output_g_cube is needed below for
  // the gradient wrt the data, and temp_ from above (input_d_cube) is not used again so it can be
  // overwritten
  // SHADJIS TODO: If in place, temp_ is used for both B and C in the GEMM -- is aliasing ok in BLAS?
  // Could make a new buffer instead.
  float* product = (in_place ? temp_ : input_g_cube)->get_p_data();

  p_driver->eltwise_mul(output_g_cube->n_elements, output_g_cube->get_p_data(), input_data, product);
  
  float* sum_result = NULL;
  if (inner_dim_ == 1) {
    sum_result = product;
  } else if (sum_result_->n_elements == 1) {
    const float* sum_mult = sum_multiplier_->get_p_data();
    float* scale_diff = p_model_gradient_cube->get_p_data();
    float result = p_driver->dot_prod(inner_dim_, product, sum_mult);
    // Should do this now:
    // *scale_diff += result;
    // But scale_diff might be on GPU, so do this instead:
    p_driver->add_scalar(1, result, scale_diff);
    // SHADJIS TODO: This only happens if iD == 1, so in that case no need for
    // a GPU array, could just always keep this scalar on the host
    // Also, I think in this code path scale_diff is uninitialized? Is that right?
  } else {
    const float* sum_mult = sum_multiplier_->get_p_data();
    sum_result = (outer_dim_ == 1) ?
        p_model_gradient_cube->get_p_data() : sum_result_->get_p_data();
    p_driver->sgemv(CblasNoTrans, sum_result_->n_elements, inner_dim_,
                   float(1), product, sum_mult, float(0), sum_result);
  }
  if (outer_dim_ != 1) {
    const float* sum_mult = sum_multiplier_->get_p_data();
    float* scale_diff = p_model_gradient_cube->get_p_data();
    if (scale_dim_ == 1) {
      float result = p_driver->dot_prod(outer_dim_, sum_mult, sum_result);
      // Should do this now:
      // *scale_diff += result;
      // But scale_diff might be on GPU, so do this instead:
      p_driver->add_scalar(1, result, scale_diff);
      // SHADJIS TODO: This only happens if iD == 1, so in that case no need for
      // a GPU array, could just always keep this scalar on the host
    } else {
      p_driver->sgemv(CblasTrans, outer_dim_, scale_dim_,
                     float(1), sum_result, sum_mult, float(true),
                     scale_diff);
    }
  }
  
  // Data gradient
  // Just like in FW pass the CPU and GPU can use the same code, but the CPU uses
  // nested loops and BLAS whereas the GPU can do it with 1 call, so use that instead
  
  // CPU
  if (std::is_same<DriverClass, CPUDriver>::value) {
    // Check in-place again, if not then copy output grad to input grad
    if (input_g_cube->get_p_data() != output_g_cube->get_p_data()) {
      p_driver->memcpy(input_g_cube->get_device_pointer(p_driver), output_g_cube->get_device_pointer(p_driver));
    }
    float* input_grad = input_g_cube->get_p_data();
    for (int n = 0; n < outer_dim_; ++n) {
      for (int d = 0; d < scale_dim_; ++d) {
        p_driver->sscale_inplace(inner_dim_, p_model_cube->get_p_data()[d], input_grad);
        input_grad += inner_dim_;
      }
    }
  }
  // GPU
  else {
    p_driver->forward_scale(iR*iC*iD*iB, output_g_cube->get_p_data(), p_model_cube->get_p_data(),
      scale_dim_, inner_dim_, input_g_cube->get_p_data());
  }  
    
  PROFILE_ONLY(p_driver->device_sync(); float seconds = t.elapsed(); std::cout << "  Bw Scale        " << seconds << "\n";)
  ////////////////////////////////////////////////////////////////////////////////

  // If this is a GPU bridge, destroy cuBLAS (does nothing on CPU)
  p_driver->destroy_thread();
  
  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
ScaleBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~ScaleBridge() {

  delete p_model_cube_shadow;
  delete p_model_cube;
  delete p_model_gradient_cube;
  if (p_bias_cube) {
    delete p_bias_cube;
  }
  if (p_bias_gradient_cube) {
    delete p_bias_gradient_cube;
  }
  delete ones_bias_vector;
  if (temp_) {
    delete temp_;
  }
  delete sum_result_;
  delete sum_multiplier_;
}

#endif
