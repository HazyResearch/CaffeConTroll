//
//  GeneralConcatBridge_impl.hxx
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef _GeneralConcatBridge_impl_hxx
#define _GeneralConcatBridge_impl_hxx

template <typename DataType, typename DriverClass>
GeneralConcatBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::GeneralConcatBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
    const cnn::SolverParameter * const _solver_param, DriverClass * const _p_driver) : AbstractBridge<DataType,
  Layout_CRDB, DataType, Layout_CRDB, DriverClass>(_p_input_layer, _p_output_layer, _layer_param, _solver_param,
      _p_driver) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();

#ifdef _DO_ASSERT
  assert(iC == oC);
  assert(iR == oR);
  assert(iB == oB);
#endif

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Forward direction for Concat
 **/
template <typename DataType, typename DriverClass>
void GeneralConcatBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::forward() {
  report_forward_last_transfer.reset();

  DataType * const output_data = p_output_layer->p_data_cube->get_p_data();
  size_t total_D_so_far = 0;
  for (size_t i_cube = 0; i_cube < p_input_layers.size(); ++i_cube){
    const DataType * const input_data = p_input_layers[i_cube]->p_data_cube->get_p_data();
    const size_t current_D = p_input_layers[i_cube]->p_data_cube->D;
    for (size_t b = 0; b < iB; ++b) {
      for (size_t r = 0; r < iR; ++r) {
        for (size_t c = 0; c < iC; ++c) {
          for (size_t d = 0; d < current_D; ++d) {
            output_data[c + r*oC + (d+total_D_so_far)*oR*oC + b*oR*oC*oD] = input_data[c + r*iC + d*iR*iC + b*iR*iC*current_D];
          }
        }
      }
    }
    total_D_so_far += current_D;
  }
#ifdef _DO_ASSERT
  assert(oD == total_D_so_far);
#endif

  report_forward_last_transfer.end();
          // TODO: iB*iD*pooled_height*pooled_width*(sizeof(DataType)+sizeof(size_t)), 0);
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Backward direction for Concat
 **/
template <typename DataType, typename DriverClass>
void GeneralConcatBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::backward() {

  report_backward_updateweight_last_transfer.reset();

  const DataType * const output_gradient = p_output_layer->p_gradient_cube->get_p_data();
  size_t total_D_so_far = 0;
  for (size_t i_cube = 0; i_cube < p_input_layers.size(); ++i_cube) {
    DataType * const input_gradient = p_input_layers[i_cube]->p_gradient_cube->get_p_data();
    const size_t current_D = p_input_layers[i_cube]->p_data_cube->D;
    for (size_t b = 0; b < iB; ++b) {
      for (size_t r = 0; r < iR; ++r) {
        for (size_t c = 0; c < iC; ++c) {
          for (size_t d = 0; d < current_D; ++d) {
            input_gradient[c + r*iC + d*iR*iC + b*iR*iC*current_D] = output_gradient[c + r*oC + (d+total_D_so_far)*oR*oC + b*oR*oC*oD];
          }
        }
      }
    }
    total_D_so_far += current_D;
  }
#ifdef _DO_ASSERT
  assert(oD == total_D_so_far);
#endif

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, typename DriverClass>
GeneralConcatBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::~GeneralConcatBridge() {
}

#endif
