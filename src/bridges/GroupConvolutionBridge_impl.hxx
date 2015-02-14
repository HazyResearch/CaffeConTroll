//
//  GroupConvolutionBridge_impl.hxx
//  moka
//
//  Created by Ce Zhang on 1/13/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_GroupConvolutionBridge_impl_hxx
#define moka_GroupConvolutionBridge_impl_hxx

// Constructor for convolution layer with group > 1
template <typename DataType, NonLinearFunction FUNC>
GroupConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB>::
GroupConvolutionBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
    const cnn::LayerParameter * const _layer_param)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_p_input_layer,
    _p_output_layer, _layer_param),
  K(layer_param->convolution_param().kernel_size()),
  num_output_features(layer_param->convolution_param().num_output()),
  stride(layer_param->convolution_param().stride()),
  padding(layer_param->convolution_param().pad()),
  bias_term(layer_param->convolution_param().bias_term()),
  weight_filler(layer_param->convolution_param().weight_filler()),
  bias_filler(layer_param->convolution_param().bias_filler()),
  grouping(layer_param->convolution_param().group()) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
  report_backward_updateweight_constructor.reset();
  report_backward_updateweight_last_transfer.reset();
  report_backward_updateweight_history.reset();

#ifdef _DO_ASSERT
  assert(grouping > 1);
  assert(num_output_features % grouping == 0);
  assert(iD % grouping == 0);
#endif

  p_model_cube = new LogicalCubeType(K, K, iD / grouping, num_output_features / grouping);
  ConvBridge::initialize_logical_cube(p_model_cube, weight_filler);

  if (bias_term) {
    p_bias_cube = new LogicalCubeType(1, 1, num_output_features, 1);
    ConvBridge::initialize_logical_cube(p_bias_cube, bias_filler);
  }

  for (size_t g = 0; g < grouping; ++g) {
    LogicalCubeType * const input_data_cube = new LogicalCubeType(NULL, iR, iC, iD / grouping, iB);
    input_data_cubes.push_back(input_data_cube);

    LogicalCubeType * const input_grad_cube = new LogicalCubeType(iR, iC, iD / grouping, iB);
    input_grad_cubes.push_back(input_grad_cube);

    LogicalCubeType * const output_data_cube = new LogicalCubeType(iR, iC, iD / grouping, iB);
    output_data_cubes.push_back(output_data_cube);

    LogicalCubeType * const output_grad_cube = new LogicalCubeType(iR, iC, iD / grouping, iB);
    output_grad_cubes.push_back(output_grad_cube);

    input_layers.push_back(
        new InputLayerType(input_data_cube, input_grad_cube)
        );

    output_layers.push_back(
        new OutputLayerType(output_data_cube, output_grad_cube)
        );
  }

  for (size_t b = 0; b < iB; ++b) {
    for (size_t d = 0; d < iD; ++d) {
    const size_t group = d / grouping;
    // TODO: partition each example along depth, and update the appropriate input data cube
    }
  }

  for (size_t g = 0; g < grouping; ++g) {
    cnn::LayerParameter * const temp_layer_param = new cnn::LayerParameter(*layer_param);
    temp_layer_param->convolution_param().set_num_output(num_output_features / grouping);
    ConvBridge * bridge = new ConvBridge(p_input_layer, p_output_layer, temp_layer_param);
    conv_bridges.push_back(bridge);
    layer_params.push_back(temp_layer_param);
  }

  report_backward_updateweight_constructor.end(0, 0, 0);
  report_forward_constructor.end(0, 0, 0);
}

/**
 * Forward pass
 **/
template <typename DataType, NonLinearFunction FUNC>
void GroupConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB>::
forward() {

  report_forward_last_transfer.reset();

  for (auto bridge = conv_bridges.begin(); bridge != conv_bridges.end(); ++bridge) {
    (*bridge)->forward();
  }

  // TODO: aggregate the individual outputs into the master output cube, p_output_layer->p_data_cube

  report_forward_last_transfer.end();
  // TODO: aggregate the report_forward_last_transfer's for each individual bridge
  // report_forward_last_transfer.aggregate_onlystat(stratum.report_forward_last_transfer);
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
  * Backward pass
 **/
template <typename DataType, NonLinearFunction FUNC>
void GroupConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB>::
backward() {

  report_backward_updateweight_last_transfer.reset();

  // TODO: split up the master output cube, p_output_layer->p_gradient_cube
  // amongst the individual bridges
  for (auto bridge = conv_bridges.rbegin(); bridge != conv_bridges.rend(); ++bridge) {
    (*bridge)->backward();
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType, NonLinearFunction FUNC>
GroupConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB>::
~GroupConvolutionBridge() {
  for (auto bridge = conv_bridges.begin(); bridge != conv_bridges.end(); ++bridge) {
    delete (*bridge);
  }

  for (auto l_p = layer_params.begin(); l_p != layer_params.end(); ++l_p) {
    delete (*l_p);
  }

  for (auto cube = input_data_cubes.begin(); cube != input_data_cubes.end(); ++cube) {
    delete (*cube);
  }

  for (auto cube = input_grad_cubes.begin(); cube != input_grad_cubes.end(); ++cube) {
    delete (*cube);
  }

  for (auto layer = input_layers.begin(); layer != input_layers.end(); ++layer) {
    delete (*layer);
  }

  for (auto cube = output_data_cubes.begin(); cube != output_data_cubes.end(); ++cube) {
    delete (*cube);
  }

  for (auto cube = output_grad_cubes.begin(); cube != output_grad_cubes.end(); ++cube) {
    delete (*cube);
  }

  for (auto layer = output_layers.begin(); layer != output_layers.end(); ++layer) {
    delete (*layer);
  }

  delete p_model_cube; delete p_bias_cube;
}

#endif
