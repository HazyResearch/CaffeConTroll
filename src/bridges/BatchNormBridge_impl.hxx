//
//  BatchNormBridge_impl.hxx
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef _BatchNormBridge_impl_hxx
#define _BatchNormBridge_impl_hxx

template <typename DataType, typename DriverClass>
BatchNormBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::BatchNormBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
    const cnn::SolverParameter * const _solver_param, DriverClass * const _p_driver)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>(_p_input_layer,
    _p_output_layer, _layer_param, _solver_param, _p_driver),
 has_use_global_stats(layer_param->batch_norm_param().has_use_global_stats()),
 use_global_stats_force(layer_param->batch_norm_param().use_global_stats()),
 moving_average_fraction_(layer_param->batch_norm_param().moving_average_fraction()),
 channels_(iD), eps_(layer_param->batch_norm_param().eps()),
 running_mean(NULL), running_variance(NULL), running_factor(0),
 mean_(NULL), variance_(NULL), temp_(NULL), x_norm_(NULL), x_norm_grad_(NULL),
 batch_sum_multiplier_(NULL), num_by_chans_(NULL), spatial_sum_multiplier_(NULL) {
 
  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(oR == iR); assert(oC == iC);
  assert(oB == iB); assert(oD == iD);
#endif
  
  // Make these 3 persistent statistics: running mean, running variance, and scale
  // Initialize to 0
  running_mean      = new LogicalCube<DataType, Layout_CRDB>(channels_, 1, 1, 1, p_driver);
  running_variance  = new LogicalCube<DataType, Layout_CRDB>(channels_, 1, 1, 1, p_driver);
  // running_factor    = new LogicalCube<DataType, Layout_CRDB>(1, 1, 1, 1, p_driver);  // float now
  p_driver->sconstant_initialize(running_mean->get_device_pointer(p_driver),     DataType(0.));
  p_driver->sconstant_initialize(running_variance->get_device_pointer(p_driver), DataType(0.));
  // p_driver->sconstant_initialize(running_factor->get_device_pointer(p_driver),   DataType(0.));
  
  // Other statistics cubes
  mean_ = new LogicalCube<DataType, Layout_CRDB>(channels_, 1, 1, 1, p_driver);   // channels_ = iD = oD
  variance_ = new LogicalCube<DataType, Layout_CRDB>(channels_, 1, 1, 1, p_driver);
  temp_ = new LogicalCube<DataType, Layout_CRDB>(iR, iC, iD, iB, p_driver);
  x_norm_ = new LogicalCube<DataType, Layout_CRDB>(iR, iC, iD, iB, p_driver);
  // If this is in place then we need a buffer to store the output grad as we are
  // changing the input grad (otherwise it will be overwritten by those changes)
  if (input_g_cube->get_p_data() == output_g_cube->get_p_data()) {
    x_norm_grad_ = new LogicalCube<DataType, Layout_CRDB>(iR, iC, iD, iB, p_driver);
  }
  batch_sum_multiplier_ = new LogicalCube<DataType, Layout_CRDB>(iB, 1, 1, 1, p_driver);

  spatial_sum_multiplier_ = new LogicalCube<DataType, Layout_CRDB>(iR*iC, 1, 1, 1, p_driver);
  p_driver->sconstant_initialize(spatial_sum_multiplier_->get_device_pointer(p_driver), DataType(1.));

  num_by_chans_ = new LogicalCube<DataType, Layout_CRDB>(channels_*iB, 1, 1, 1, p_driver);
  
  p_driver->sconstant_initialize(batch_sum_multiplier_->get_device_pointer(p_driver), DataType(1.));

  report_forward_constructor.end(0, 0, 0);
}


template <typename DataType, typename DriverClass>
void BatchNormBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  output_d_cube->set_p_data(p_output_layer->p_data_cube->get_p_data());
  input_d_cube ->set_p_data(p_input_layer ->p_data_cube->get_p_data());

  report_forward_last_transfer.reset();

  // If this is a GPU bridge, init cuBLAS (does nothing on CPU)
  p_driver->init_thread();
  
  ////////////////////////////////////////////////////////////////////////////////
  
  PROFILE_ONLY(Timer t; Timer t_inner; float seconds;)

  bool use_global_stats_ = !DeepNetConfig::train();  // global in test only
  if (has_use_global_stats) {
    use_global_stats_ = use_global_stats_force;
  }
  
  DeviceMemoryPointer * input = input_d_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = output_d_cube->get_device_pointer(p_driver);

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    1:  " << seconds << "\n"; t_inner.restart();)

  if (input_d_cube->get_p_data() != output_d_cube->get_p_data()) {
    p_driver->memcpy(output, input);
  }

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    2:  " << seconds << "\n"; t_inner.restart();)

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const float scale_factor = running_factor == 0 ? 0 : 1./running_factor;
    p_driver->sscale(variance_->n_elements, scale_factor,
        running_mean->get_p_data(), mean_->get_p_data());
    p_driver->sscale(variance_->n_elements, scale_factor,
        running_variance->get_p_data(), variance_->get_p_data());
  } else {
  
    // compute mean
    p_driver->sgemv(CblasNoTrans, channels_ * iB, iR*iC,
        1. / (iB * iR*iC), input_d_cube->get_p_data(),
        spatial_sum_multiplier_->get_p_data(), 0.,
        num_by_chans_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    3:  " << seconds << "\n"; t_inner.restart();)
        
    p_driver->sgemv(CblasTrans, iB, channels_, 1.,
        num_by_chans_->get_p_data(), batch_sum_multiplier_->get_p_data(), 0.,
        mean_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    4:  " << seconds << "\n"; t_inner.restart();)
        
  }

  // subtract mean
  p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, iB, channels_, 1, 1,
      batch_sum_multiplier_->get_p_data(), mean_->get_p_data(), 0.,
      num_by_chans_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    5:  " << seconds << "\n"; t_inner.restart();)
        
  p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, channels_ * iB,
      iR*iC, 1, -1, num_by_chans_->get_p_data(),
      spatial_sum_multiplier_->get_p_data(), 1., output_d_cube->get_p_data());


  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    6:  " << seconds << "\n"; t_inner.restart();)
        
  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    p_driver->eltwise_pow2(iR*iC*iD*iB, output_d_cube->get_p_data(),
        temp_->get_p_data());  // (X-EX)^2

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    7:  " << seconds << "\n"; t_inner.restart();)
        
    p_driver->sgemv(CblasNoTrans, channels_ * iB, iR*iC,
        1. / (iB * iR*iC), temp_->get_p_data(),
        spatial_sum_multiplier_->get_p_data(), 0.,
        num_by_chans_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    8:  " << seconds << "\n"; t_inner.restart();)
        
    p_driver->sgemv(CblasTrans, iB, channels_, 1.,
        num_by_chans_->get_p_data(), batch_sum_multiplier_->get_p_data(), 0.,
        variance_->get_p_data());  // E((X_EX)^2)

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    9:  " << seconds << "\n"; t_inner.restart();)
        
    // compute and save moving average
    running_factor *= moving_average_fraction_;
    running_factor += 1;
    p_driver->math_saxpby(mean_->n_elements, float(1), mean_->get_p_data(),
        moving_average_fraction_, running_mean->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   10:  " << seconds << "\n"; t_inner.restart();)
        
    int m = iB*iR*iC;
    float bias_correction_factor = m > 1 ? float(m)/(m-1) : 1;
    p_driver->math_saxpby(variance_->n_elements, bias_correction_factor,
        variance_->get_p_data(), moving_average_fraction_,
        running_variance->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   11:  " << seconds << "\n"; t_inner.restart();)
        
  }

  // normalize variance
  p_driver->add_scalar(variance_->n_elements, eps_, variance_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   12:  " << seconds << "\n"; t_inner.restart();)
        
  p_driver->eltwise_sqrt(variance_->n_elements, variance_->get_p_data(),
            variance_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   13:  " << seconds << "\n"; t_inner.restart();)
        
  // replicate variance to input size
  p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, iB, channels_, 1, 1,
      batch_sum_multiplier_->get_p_data(), variance_->get_p_data(), 0.,
      num_by_chans_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   14:  " << seconds << "\n"; t_inner.restart();)
        
  p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, channels_ * iB,
      iR*iC, 1, 1., num_by_chans_->get_p_data(),
      spatial_sum_multiplier_->get_p_data(), 0., temp_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   15:  " << seconds << "\n"; t_inner.restart();)
        
  p_driver->eltwise_div(temp_->n_elements, output_d_cube->get_p_data(), temp_->get_p_data(), output_d_cube->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   16:  " << seconds << "\n"; t_inner.restart();)
        
  // This and LRN are the only bridges which need the output data in the backwards pass.
  // We need to backup the output data therefore, because future bridges might be in-place.
  // Scale is similar -- it needs the input data in the backwards pass, so if it was in-place 
  // the input was backed up in a temp buffer.
  p_driver->memcpy(x_norm_->get_device_pointer(p_driver), output_d_cube->get_device_pointer(p_driver));

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   17:  " << seconds << "\n"; t_inner.restart();)
        
  PROFILE_ONLY(seconds = t.elapsed(); std::cout << "Fw Total:  " << seconds << "\n";)
        
  ////////////////////////////////////////////////////////////////////////////////

  // If this is a GPU bridge, destroy cuBLAS (does nothing on CPU)
  p_driver->destroy_thread();
  
  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}


template <typename DataType, typename DriverClass>
void BatchNormBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  input_g_cube ->set_p_data(p_input_layer ->p_gradient_cube->get_p_data());
  output_g_cube->set_p_data(p_output_layer->p_gradient_cube->get_p_data());

  report_backward_updateweight_last_transfer.reset();

  // If this is a GPU bridge, init cuBLAS (does nothing on CPU)
  p_driver->init_thread();
  
  ////////////////////////////////////////////////////////////////////////////////

  PROFILE_ONLY(Timer t; Timer t_inner; float seconds;)
  
  bool use_global_stats_ = !DeepNetConfig::train();  // global in test only
  if (has_use_global_stats) {
    use_global_stats_ = use_global_stats_force;
  }
  
  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    1:  " << seconds << "\n"; t_inner.restart();)

  const float* output_grad;
  if (input_g_cube->get_p_data() != output_g_cube->get_p_data()) {
    output_grad = output_g_cube->get_p_data();
  } else {
    p_driver->memcpy(x_norm_grad_->get_device_pointer(p_driver), output_g_cube->get_device_pointer(p_driver));
    output_grad = x_norm_grad_->get_p_data();
  }

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    2:  " << seconds << "\n"; t_inner.restart();)

  float * input_grad = input_g_cube->get_p_data();
  if (use_global_stats_) {
    p_driver->eltwise_div(temp_->n_elements, output_grad, temp_->get_p_data(), input_grad);
    return;
  }

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    3:  " << seconds << "\n"; t_inner.restart();)

  const float* output_data = x_norm_->get_p_data();
  int num = iB;
  int spatial_dim = iR*iC;
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the output grad, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  p_driver->eltwise_mul(temp_->n_elements, output_data, output_grad, input_grad);

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    4:  " << seconds << "\n"; t_inner.restart();)

  p_driver->sgemv(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      input_grad, spatial_sum_multiplier_->get_p_data(), 0.,
      num_by_chans_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    5:  " << seconds << "\n"; t_inner.restart();)

  p_driver->sgemv(CblasTrans, num, channels_, 1.,
      num_by_chans_->get_p_data(), batch_sum_multiplier_->get_p_data(), 0.,
      mean_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    6:  " << seconds << "\n"; t_inner.restart();)

  // reshape (broadcast) the above
  p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_->get_p_data(), mean_->get_p_data(), 0.,
      num_by_chans_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    7:  " << seconds << "\n"; t_inner.restart();)

  p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_->get_p_data(),
      spatial_sum_multiplier_->get_p_data(), 0., input_grad);

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    8:  " << seconds << "\n"; t_inner.restart();)

  // sum(dE/dY \cdot Y) \cdot Y
  p_driver->eltwise_mul(temp_->n_elements, output_data, input_grad, input_grad);

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "    9:  " << seconds << "\n"; t_inner.restart();)

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  p_driver->sgemv(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      output_grad, spatial_sum_multiplier_->get_p_data(), 0.,
      num_by_chans_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   10:  " << seconds << "\n"; t_inner.restart();)

  p_driver->sgemv(CblasTrans, num, channels_, 1.,
      num_by_chans_->get_p_data(), batch_sum_multiplier_->get_p_data(), 0.,
      mean_->get_p_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   11:  " << seconds << "\n"; t_inner.restart();)

  p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_->get_p_data(), mean_->get_p_data(), 0.,
      num_by_chans_->get_p_data());

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   12:  " << seconds << "\n"; t_inner.restart();)

  p_driver->sgemm_new(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_->get_p_data(),
      spatial_sum_multiplier_->get_p_data(), 1., input_grad);

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   13:  " << seconds << "\n"; t_inner.restart();)

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  p_driver->math_saxpby(temp_->n_elements, float(1), output_grad,
      float(-1. / (num * spatial_dim)), input_grad);

  PROFILE_ONLY(seconds = t_inner.elapsed(); std::cout << "   14:  " << seconds << "\n"; t_inner.restart();)

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  p_driver->eltwise_div(temp_->n_elements, input_grad, temp_->get_p_data(), input_grad);

  PROFILE_ONLY(seconds = t.elapsed(); std::cout << "Bw Total:  " << seconds << "\n";)
        
  ////////////////////////////////////////////////////////////////////////////////

  // If this is a GPU bridge, destroy cuBLAS (does nothing on CPU)
  p_driver->destroy_thread();

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
BatchNormBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~BatchNormBridge() {

  delete running_mean;
  delete running_variance;
  
  delete mean_;
  delete variance_;
  delete temp_;
  delete x_norm_;
  if (x_norm_grad_) {
    delete x_norm_grad_;
  }
  
  delete batch_sum_multiplier_;
  delete spatial_sum_multiplier_;
  delete num_by_chans_;
}

#endif
