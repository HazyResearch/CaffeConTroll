//
//  FunnelBridge_impl.hxx
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_FunnelBridge_impl_hxx
#define moka_FunnelBridge_impl_hxx

template <typename DataType, typename DriverClass>
FunnelBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::FunnelBridge(InputLayerType * const _p_input_layer,
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

// SHADJIS TODO: This now assumes iD is same for all bridges, but we have p_input_layers
// so each cube can have different depth (e.g. for CPU + GPU model parallelism by FLOPS) 

/**
 * Forward direction for Funnel
 **/
template <typename DataType, typename DriverClass>
void FunnelBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
  report_forward_last_transfer.reset();

  // SHADJIS TODO: Not here we assume that each layer in p_input_layers has the
  // same cube sizes, and moreover that they match iR/iC/iD/iB which came from
  // p_input_layer.
  DataType * const output_data = p_output_layer->p_data_cube->get_p_data();
  for (size_t i_cube = 0; i_cube < p_input_layers.size(); ++i_cube){
    const DataType * const input_data = p_input_layers[i_cube]->p_data_cube->get_p_data();
    for (size_t b = 0; b < iB; ++b) {
      for (size_t r = 0; r < iR; ++r) {
        for (size_t c = 0; c < iC; ++c) {
          for (size_t d = 0; d < iD; ++d) {
            output_data[c + r*oC + (d+iD*i_cube)*oR*oC + b*oR*oC*oD] = input_data[c + r*iC + d*iR*iC + b*iR*iC*iD];
          }
        }
      }
    }
  }

  report_forward_last_transfer.end();
          // TODO: iB*iD*pooled_height*pooled_width*(sizeof(DataType)+sizeof(size_t)), 0);
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Backward direction for Funnel
 **/
template <typename DataType, typename DriverClass>
void FunnelBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  report_backward_updateweight_last_transfer.reset();

  // SHADJIS TODO: Not here we assume that each layer in p_input_layers has the
  // same cube sizes, and moreover that they match iR/iC/iD/iB which came from
  // p_input_layer.
  const DataType * const output_gradient = p_output_layer->p_gradient_cube->get_p_data();
  for (size_t i_cube = 0; i_cube < p_input_layers.size(); ++i_cube) {
    DataType * const input_gradient = p_input_layers[i_cube]->p_gradient_cube->get_p_data();
    for (size_t b = 0; b < iB; ++b) {
      for (size_t r = 0; r < iR; ++r) {
        for (size_t c = 0; c < iC; ++c) {
          for (size_t d = 0; d < iD; ++d) {
            input_gradient[c + r*iC + d*iR*iC + b*iR*iC*iD] = output_gradient[c + r*oC + (d+iD*i_cube)*oR*oC + b*oR*oC*oD];
          }
        }
      }
    }
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
FunnelBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~FunnelBridge() {
}

#endif
