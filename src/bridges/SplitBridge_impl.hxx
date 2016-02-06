//
//  SplitBridge_impl.hxx
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef _SplitBridge_impl_hxx
#define _SplitBridge_impl_hxx

template <typename DataType, typename DriverClass>
SplitBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::SplitBridge(InputLayerType * const _p_input_layer,
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
 * Forward direction for Split
 **/
template <typename DataType, typename DriverClass>
void SplitBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
  report_forward_last_transfer.reset();
  
#ifdef _DO_ASSERT
  for (size_t i = 0; i < p_output_layers.size(); ++i) {
    assert(p_output_layers[i]->p_data_cube->get_p_data() == p_input_layer->p_data_cube->get_p_data());
  }
#endif

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Backward direction for Split
 **/
template <typename DataType, typename DriverClass>
void SplitBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  report_backward_updateweight_last_transfer.reset();

  // Minimize # writes for special-cases
  // Can support more cases as well. Unlike funnel, split does not change depth.
  float * const output = p_input_layer->p_gradient_cube->get_p_data();
  const size_t n_elements = p_input_layer->p_gradient_cube->n_elements;
  if (p_output_layers.size() == 4) {
    const float * const input0 = p_output_layers[0]->p_gradient_cube->get_p_data();
    const float * const input1 = p_output_layers[1]->p_gradient_cube->get_p_data();
    const float * const input2 = p_output_layers[2]->p_gradient_cube->get_p_data();
    const float * const input3 = p_output_layers[3]->p_gradient_cube->get_p_data();
    for (size_t i = 0; i < n_elements; ++i) {
      output[i] = input0[i] + input1[i] + input2[i] + input3[i];
    }
  }
  else if (p_output_layers.size() == 2) {
    const float * const input0 = p_output_layers[0]->p_gradient_cube->get_p_data();
    const float * const input1 = p_output_layers[1]->p_gradient_cube->get_p_data();
    for (size_t i = 0; i < n_elements; ++i) {
      output[i] = input0[i] + input1[i];
    }
  }
  // General case
  // SHADJIS TODO: If bottleneck, reorder loops to minimize writes
  else {
    p_input_layer->p_gradient_cube->reset_cube();
    // Iterate over each output cube's gradient and sum it to the input cube's gradient
    for (size_t output_grad_cube_idx = 0; output_grad_cube_idx < p_output_layers.size(); ++output_grad_cube_idx) {
      const float * const input = p_output_layers[output_grad_cube_idx]->p_gradient_cube->get_p_data();
      for (size_t i = 0; i < n_elements; ++i) {
        output[i] += input[i];
      }
    }
  }
  
  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
SplitBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~SplitBridge() {
}

#endif
