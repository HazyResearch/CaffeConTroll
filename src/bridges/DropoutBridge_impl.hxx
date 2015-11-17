//
//  DropoutBridge_impl.hxx
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_DropoutBridge_impl_hxx
#define moka_DropoutBridge_impl_hxx

template <typename DataType, typename DriverClass>
DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::DropoutBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
    const cnn::SolverParameter * const _solver_param, DriverClass * const _p_driver)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>(_p_input_layer, _p_output_layer,
    _layer_param, _solver_param, _p_driver), dropout_ratio(layer_param->dropout_param().dropout_ratio()),
    // SHADJIS TODO: These remaining members are for the bernoulli generation on the CPU
    // This should be part of the device driver instead, but I noticed that was slightly slower
    // Need to profile again though. Then call and call p_driver->sbernoulli_initialize in fw pass.
    rng(mt19937(rd())), random_distribution(boost::bernoulli_distribution<float>(1. - dropout_ratio)), 
    variate_generator(boost::variate_generator<mt19937, boost::bernoulli_distribution<float> >(rng, random_distribution)) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(oR == iR); assert(oC == iC);
  assert(oB == iB); assert(oD == iD);
#ifndef _SNAPSHOT
  assert(dropout_ratio > 0.);
#endif
  assert(dropout_ratio < 1.);
#endif

  scale = 1. / (1. - dropout_ratio);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * dropout_ratio);
  
  // Mask cube -- note this memory is allocated on the device
  mask_cube = new LogicalCube<unsigned int, Layout_CRDB>(iR, iC, iD, iB, p_driver);

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Implements Dropout in the forward direction.
 **/
template <typename DataType, typename DriverClass>
void DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  input_d_cube ->set_p_data(p_input_layer ->p_data_cube->get_p_data());
  output_d_cube->set_p_data(p_output_layer->p_data_cube->get_p_data());

  report_forward_last_transfer.reset();
#ifdef _DO_ASSERT
  assert(p_input_layer->p_data_cube->n_elements == mask_cube->n_elements);
#endif

  ////////////////////////////////////////////////////////////////////////////////
  PROFILE_ONLY(p_driver->device_sync(); Timer t;)
  
  // In the training phase, we apply the mask
  if (DeepNetConfig::train()) {
  
    // CPU version -- generate and apply mask on CPU
    if (std::is_same<DriverClass, CPUDriver>::value) {
      // Create a new mask for this iteration
      // SHADJIS TODO: This code should be refactored into p_driver->sbernoulli_initialize but for now I noticed that is 2ms slower 
      // for dropout6 and 7 fw (7ms -> 9ms), probably because of device memory pointers? Tested on g2.8xlarge
      unsigned int * const mask = mask_cube->get_p_data();
      for (size_t i=0; i < mask_cube->n_elements; ++i) {
        mask[i] = variate_generator();
      }
      const float * const input = input_d_cube->get_p_data();
      float * const output = output_d_cube->get_p_data();
      const float const_scale = scale;
      for (size_t i=0; i < mask_cube->n_elements; ++i) {
        output[i] = input[i] * mask[i] * const_scale;
      }
    }
    // GPU version -- generate and apply mask on GPU
    else {
     
      // On the GPU there is an optimization to share input/output layers for
      // this and ReLU, so we know that the input and output layers match.
      // This can simplify the dropout layer a bit by re-using a pointer (which
      // isn't any faster?) and also skipping the test phase (since dropout
      // in test mode is just a copy, but this doesn't make the code any faster
      // since testing is fast and copies are too)
#ifdef _DO_ASSERT
      assert(input_d_cube->get_p_data() == output_d_cube->get_p_data());
#endif
     
      // Create a new mask for this iteration
      p_driver->rand_uint_initialize(mask_cube->get_p_data(), mask_cube->n_elements);
      _dropout_forward_train_arg_helper _arg;
      _arg.mask = (char *) mask_cube->get_p_data();
      _arg.scale = scale;
      _arg.threshold = uint_thres_;
     
      DeviceMemoryPointer * input_and_output = input_d_cube->get_device_pointer(p_driver);
      DeviceMemoryPointer * arg1 = p_driver->get_device_pointer(NULL, 0);
      DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg,
        sizeof(_dropout_forward_train_arg_helper));

      p_driver->template parallel_map<_f_src_to_dst_dropout_forward,
        _f_dropout_forward_train>(input_and_output, input_and_output, sizeof(DataType), arg1, arg2);
    }
  }
  // In the testing phase, we just need to copy from input to output
  // Since the pointers are shared on the GPU, we don't have to do anything
  // For the CPU, this is just a memcpy
  else if (std::is_same<DriverClass, CPUDriver>::value) {
    memcpy(output_d_cube->get_p_data(), input_d_cube->get_p_data(), sizeof(float)*output_d_cube->n_elements);
  }

  PROFILE_ONLY(p_driver->device_sync(); float seconds = t.elapsed(); std::cout << "  Fw Dropout        " << seconds << "\n";)
  ////////////////////////////////////////////////////////////////////////////////

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}

/**
 * Implements Dropout in the backward direction.
 **/
template <typename DataType, typename DriverClass>
void DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  // Make sure the internal cube pointers of this abstract bridge match the bridge's layer cubes
  output_g_cube->set_p_data(p_output_layer->p_gradient_cube->get_p_data());
  input_g_cube ->set_p_data(p_input_layer ->p_gradient_cube->get_p_data());

  report_backward_updateweight_last_transfer.reset();
#ifdef _DO_ASSERT
  assert(p_input_layer->p_gradient_cube->n_elements == mask_cube->n_elements);
#endif

  ////////////////////////////////////////////////////////////////////////////////
  PROFILE_ONLY(p_driver->device_sync(); Timer t;)
  
  // CPU version -- apply mask on CPU
  if (std::is_same<DriverClass, CPUDriver>::value) {
    // Use mask from fw pass
    const unsigned int * const  mask = mask_cube->get_p_data();
    const float * const output = output_g_cube->get_p_data();
    float * const input = input_g_cube->get_p_data(); // In backwards pass, we write to input
    const float const_scale = scale;
    for (size_t i=0; i < mask_cube->n_elements; ++i) {
      input[i] = output[i] * mask[i] * const_scale;
    }
  }
  // GPU version -- apply mask on GPU
  else {

    // On the GPU there is an optimization to share input/output layers for
    // this and ReLU, so we know that the input and output layers match.
    // This can simplify the dropout layer a bit by re-using a pointer (which
    // isn't any faster?) and also skipping the test phase (since dropout
    // in test mode is just a copy, but this doesn't make the code any faster
    // since testing is fast and copies are too)
#ifdef _DO_ASSERT
    assert(input_g_cube->get_p_data() == output_g_cube->get_p_data());
#endif
   
    _dropout_forward_train_arg_helper _arg;
    _arg.mask = (char *) mask_cube->get_p_data();
    _arg.scale = scale;
    _arg.threshold = uint_thres_;
   
    DeviceMemoryPointer * input_and_output = input_g_cube->get_device_pointer(p_driver);
    DeviceMemoryPointer * arg1 = p_driver->get_device_pointer(NULL, 0);
    DeviceMemoryPointer * arg2 = p_driver->get_device_pointer((void*)&_arg,
        sizeof(_dropout_forward_train_arg_helper));
   
    // the backward phase is the same as the forward phase, except we treat
    // the input gradient as the output, and the output gradient as the input
    p_driver->template parallel_map<_f_src_to_dst_dropout_forward,
      _f_dropout_forward_train>(input_and_output, input_and_output, sizeof(DataType), arg1, arg2);
  }

  PROFILE_ONLY(p_driver->device_sync(); float seconds = t.elapsed(); std::cout << "  Bw Droput        " << seconds << "\n";)
  ////////////////////////////////////////////////////////////////////////////////

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~DropoutBridge() {
  delete mask_cube;
}

#endif
