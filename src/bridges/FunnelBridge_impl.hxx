//
//  MaxPoolingBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/25/15.
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

/**
 * Forward direction for Funnel
 **/
template <typename DataType, typename DriverClass>
void FunnelBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
  report_forward_last_transfer.reset();

  // TODO: This is ugly in the following ways:
  //    - p_input_layers should be pushed from the constructor
  size_t real_depth;
  for (int i_cube = 0; i_cube < p_input_layers.size(); i_cube ++){
    LogicalCube<DataType, Layout_CRDB> * real_input_cube = p_input_layers[i_cube]->p_data_cube;
    for (size_t b_i = 0; b_i < iB; ++b_i) {
      for (size_t d_i = 0; d_i < iD; ++d_i) {
        real_depth = d_i + iD*i_cube;
        for (size_t r_i = 0; r_i < iR; ++r_i) {
          for (size_t c_i = 0; c_i < iC; ++c_i) {
            *(p_output_layer->p_data_cube->logical_get(r_i, c_i, real_depth, b_i)) =
               *(real_input_cube->logical_get(r_i, c_i, d_i, b_i));
          }
        }
      }
    }
  }

  report_forward_last_transfer.end(1.0*iB*iD*iR*iC*sizeof(DataType),
          iB*iD*(sizeof(DataType)+sizeof(size_t)), 0);
          // TODO: iB*iD*pooled_height*pooled_width*(sizeof(DataType)+sizeof(size_t)), 0);
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Backward direction for Funnel
 **/
template <typename DataType, typename DriverClass>
void FunnelBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  report_backward_updateweight_last_transfer.reset();

  p_input_layer->p_gradient_cube->reset_cube();

  size_t real_depth;
  for (int i_cube = 0; i_cube < p_input_layers.size(); i_cube ++) {
    LogicalCube<DataType, Layout_CRDB> * real_input_cube = p_input_layers[i_cube]->p_gradient_cube;
    for (size_t b_i = 0; b_i < iB; ++b_i) {
      for (size_t d_i = 0; d_i < iD; ++d_i) {
        real_depth = d_i + iD*i_cube;
        for (size_t r_i = 0; r_i < iR; ++r_i) {
          for (size_t c_i = 0; c_i < iC; ++c_i) {
            *(real_input_cube->logical_get(r_i, c_i, d_i, b_i)) =
              *(p_output_layer->p_gradient_cube->logical_get(r_i, c_i, real_depth, b_i));
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
