//
//  ParallelizedConvolutionBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 2/8/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_ParallelizedConvolutionBridge_impl_hxx
#define moka_ParallelizedConvolutionBridge_impl_hxx


// common initialization code, called by both constructors
template<typename DataType>
void ParallelizedConvolutionBridge<DataType>::initialize() {
  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
  report_backward_updateweight_constructor.reset();
  report_backward_updateweight_last_transfer.reset();
  report_backward_updateweight_history.reset();

  // Right now, we only parititon by data, not model
  for (size_t b = 0; b < n_batch; b += n_batch_per_partition) {

    const size_t n_batch_this_partition = b + n_batch_per_partition
      >= n_batch ? n_batch - b : n_batch_per_partition;

    _data_cubes_lower.push_back(
        new LogicalCubeType(NULL, //p_input_layer->p_data_cube->physical_get_RCDslice(b),
          p_input_layer->dR, p_input_layer->dC, p_input_layer->dD, n_batch_this_partition)
        );

    _grad_cubes_lower.push_back(
        new LogicalCubeType(p_input_layer->p_gradient_cube->physical_get_RCDslice(b),
          p_input_layer->gR, p_input_layer->gC, p_input_layer->gD, n_batch_this_partition)
        );

    _data_cubes_higher.push_back(
        new LogicalCubeType(p_output_layer->p_data_cube->physical_get_RCDslice(b),
          p_output_layer->dR, p_output_layer->dC, p_output_layer->dD, n_batch_this_partition)
        );

    _grad_cubes_higher.push_back(
        new LogicalCubeType(p_output_layer->p_gradient_cube->physical_get_RCDslice(b),
          p_output_layer->gR, p_output_layer->gC, p_output_layer->gD, n_batch_this_partition)
        );
  }

  for (size_t ib = 0; ib < _data_cubes_lower.size(); ib++) {
    _partitioned_layers_lower.push_back(
        new LayerType(_data_cubes_lower[ib], _grad_cubes_lower[ib])
        );

    _partitioned_layers_higher.push_back(
        new LayerType(_data_cubes_higher[ib], _grad_cubes_higher[ib])
        );
  }

  for (size_t ib = 0; ib < _data_cubes_lower.size(); ib++) {
    if (layer_param) {
      _bridges.push_back(
          new ConvolutionBridgeType(_partitioned_layers_lower[ib], _partitioned_layers_higher[ib],
            layer_param)
          );
    } else if (config) {
      _bridges.push_back(
          new ConvolutionBridgeType(_partitioned_layers_lower[ib], _partitioned_layers_higher[ib],
            config)
          );
    } else {
      cout << "ERROR! Both layer_param and config are NULL" << endl;
      assert(false);
    }
  }

  for (size_t ib = 0; ib < _data_cubes_lower.size(); ib++) {
    _bridges[ib]->run_with_n_threads = n_thread_per_partition;
    stratum.executors.push_back((PhysicalOperator *)_bridges[ib]);
  }

  report_backward_updateweight_constructor.end(0, 0, 0);
  report_forward_constructor.end(0, 0, 0);
}

// Network initialization constructor
template<typename DataType>
ParallelizedConvolutionBridge<DataType>::ParallelizedConvolutionBridge(Layer<DataType,Layout_CRDB> * const _input_layer,
    Layer<DataType, Layout_CRDB> * const _output_layer, const cnn::LayerParameter * const _layer_param, size_t _n_partition,
    size_t _n_thread_per_partition) : AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_input_layer,
      _output_layer, _layer_param), config(NULL), n_partition(_n_partition), n_batch(_input_layer->dB),
n_thread_per_partition(_n_thread_per_partition), n_batch_per_partition(n_batch / n_partition) {
  initialize();
}

// Testing constructor
template<typename DataType>
ParallelizedConvolutionBridge<DataType>::ParallelizedConvolutionBridge(Layer<DataType,Layout_CRDB> * const _input_layer,
    Layer<DataType, Layout_CRDB> * const _output_layer, const BridgeConfig * const _config, size_t _n_partition,
    size_t _n_thread_per_partition) : AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_input_layer,
      _output_layer), config(_config), n_partition(_n_partition), n_batch(_input_layer->dB),
n_thread_per_partition(_n_thread_per_partition), n_batch_per_partition(n_batch / n_partition) {
  initialize();
}

template<typename DataType>
void ParallelizedConvolutionBridge<DataType>::forward() {
  report_forward_last_transfer.reset();
  for (size_t i = 0; i < _data_cubes_lower.size(); ++i) {
    _data_cubes_lower[i]->p_data = p_input_layer->p_data_cube->physical_get_RCDslice(i*n_batch_per_partition);
  }
  stratum.forward();
  report_forward_last_transfer.end();
  report_forward_last_transfer.aggregate_onlystat(stratum.report_forward_last_transfer);
  report_forward_history.aggregate(report_forward_last_transfer);
}

template<typename DataType>
void ParallelizedConvolutionBridge<DataType>::backward() {
  report_backward_updateweight_last_transfer.reset();
  stratum.backward();
  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_last_transfer.aggregate_onlystat(stratum.report_backward_updateweight_last_transfer);
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

#endif
