//
//  EltwiseBridge_impl.hxx
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef _EltwiseBridge_impl_hxx
#define _EltwiseBridge_impl_hxx

template <typename DataType, typename DriverClass>
EltwiseBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::EltwiseBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
    const cnn::SolverParameter * const _solver_param, DriverClass * const _p_driver) : AbstractBridge<DataType,
  Layout_CRDB, DataType, Layout_CRDB, DriverClass>(_p_input_layer, _p_output_layer, _layer_param, _solver_param,
      _p_driver) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();

#ifdef _DO_ASSERT
  assert(iB == oB);
#endif

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Forward direction for Eltwise
 **/
template <typename DataType, typename DriverClass>
void EltwiseBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
  report_forward_last_transfer.reset();

  // Minimize # writes for special-cases
  float * const output = p_output_layer->p_data_cube->get_p_data();
  const size_t n_elements = p_output_layer->p_data_cube->n_elements;
  if (p_input_layers.size() == 4) {
    const float * const input0 = p_input_layers[0]->p_data_cube->get_p_data();
    const float * const input1 = p_input_layers[1]->p_data_cube->get_p_data();
    const float * const input2 = p_input_layers[2]->p_data_cube->get_p_data();
    const float * const input3 = p_input_layers[3]->p_data_cube->get_p_data();
    for (size_t i = 0; i < n_elements; ++i) {
      output[i] = input0[i] + input1[i] + input2[i] + input3[i];
    }
  }
  else if (p_input_layers.size() == 2) {
    const float * const input0 = p_input_layers[0]->p_data_cube->get_p_data();
    const float * const input1 = p_input_layers[1]->p_data_cube->get_p_data();
    for (size_t i = 0; i < n_elements; ++i) {
      output[i] = input0[i] + input1[i];
    }
  }
  // General case
  // SHADJIS TODO: If bottleneck, reorder loops to minimize writes
  else {
    p_output_layer->p_data_cube->reset_cube(); // SHADJIS TODO: won't work on device
    // Iterate over each output cube's gradient and sum it to the input cube's gradient
    for (size_t input_data_cube_idx = 0; input_data_cube_idx < p_input_layers.size(); ++input_data_cube_idx) {
      const float * const input = p_input_layers[input_data_cube_idx]->p_data_cube->get_p_data();
      for (size_t i = 0; i < n_elements; ++i) {
        output[i] += input[i];
      }
    }
  }

  report_forward_last_transfer.end();
          // TODO: iB*iD*pooled_height*pooled_width*(sizeof(DataType)+sizeof(size_t)), 0);
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Backward direction for Eltwise
 **/
template <typename DataType, typename DriverClass>
void EltwiseBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  report_backward_updateweight_last_transfer.reset();

#ifdef _DO_ASSERT
  for (size_t i = 0; i < p_input_layers.size(); ++i) {
    assert(p_input_layers[i]->p_gradient_cube->get_p_data() == p_output_layer->p_gradient_cube->get_p_data());
  }
#endif

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
EltwiseBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~EltwiseBridge() {
}

#endif
