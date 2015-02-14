//
//  ParallelizedBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 2/8/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_ParallelizedBridge_impl_hxx
#define moka_ParallelizedBridge_impl_hxx

template<typename DataType, typename BridgeType>
ParallelizedBridge<DataType, BridgeType>::ParallelizedBridge(Layer<DataType,Layout_CRDB> * const _input_layer,
    Layer<DataType, Layout_CRDB> * const _output_layer, const cnn::LayerParameter * const _layer_param, size_t _n_partition,
    size_t _n_thread_per_partition) : AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_input_layer,
      _output_layer, _layer_param), n_partition(_n_partition), n_batch(_input_layer->dB),
n_thread_per_partition(_n_thread_per_partition), n_batch_per_partition(n_batch / n_partition) {

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
        new LogicalCubeType(NULL, p_input_layer->dR, p_input_layer->dC,
          p_input_layer->dD, n_batch_this_partition)
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
    _bridges.push_back(
        new BridgeType(_partitioned_layers_lower[ib], _partitioned_layers_higher[ib], layer_param)
        );
  }

  for (size_t ib = 0; ib < _data_cubes_lower.size(); ib++) {
    _bridges[ib]->run_with_n_threads = n_thread_per_partition;
    stratum.executors.push_back((PhysicalOperator *)_bridges[ib]);
  }


  // create the master model for this parallel bridge
  assert(_bridges.size() >= 1); // we need at least one partition.
  BridgeType * const example_bridge = _bridges[0]; // get one convbrdige as example
  // TODO: p_model_cube should be T * __const__ -- but this involes changing the
  // constructor, need to discuss with Firas in detials
  LogicalCubeType * const examle_cube = _bridges[0]->get_model_cube();
  if (examle_cube != NULL) {
    p_model_cube = new LogicalCubeType(examle_cube->R, examle_cube->C, examle_cube->D, examle_cube->B);
      memcpy(p_model_cube->p_data, examle_cube->p_data, p_model_cube->n_elements*sizeof(DataType));
  } else {
    p_model_cube = NULL;
  }

  if (example_bridge->bias_term) {
    LogicalCubeType * const example_bias = _bridges[0]->get_bias_cube();
    if (example_bias != NULL) {
      p_bias_cube = new LogicalCubeType(example_bias->R, example_bias->C, example_bias->D, example_bias->B);
      memcpy(p_bias_cube->p_data, example_bias->p_data, p_bias_cube->n_elements*sizeof(DataType));
    } else {
      p_bias_cube = NULL;
    }
  }

  report_backward_updateweight_constructor.end(0, 0, 0);
  report_forward_constructor.end(0, 0, 0);
}

template<typename DataType, typename BridgeType>
void ParallelizedBridge<DataType, BridgeType>::forward() {

  report_forward_last_transfer.reset();
  for (size_t i = 0; i < _data_cubes_lower.size(); ++i) {
    _data_cubes_lower[i]->p_data = p_input_layer->p_data_cube->physical_get_RCDslice(i*n_batch_per_partition);
    // we also need to copy the mode around
    // TODO: this could be optimized by sharing model pointer across all worker -- this is
    // not implemented for now because it involves changing single-partition ConvBridge's interface.
    // so need more careful review with Firas before changing.
    //    Also, given GEMM reads the model once anyway and is still CPU bound, the current performance
    // should not be terrible

    _bridges[i]->set_model_cube(p_model_cube);
    _bridges[i]->set_bias_cube(p_bias_cube);

    /*
    memcpy(_bridges[i]->model_cube()->p_data, p_model_cube->p_data, p_model_cube->n_elements*sizeof(DataType));
    memcpy(_bridges[i]->bias_cube()->p_data, p_bias_cube->p_data, p_bias_cube->n_elements*sizeof(DataType));
    */
  }
  stratum.forward();
  report_forward_last_transfer.end();
  report_forward_last_transfer.aggregate_onlystat(stratum.report_forward_last_transfer);
  report_forward_history.aggregate(report_forward_last_transfer);
}

template<typename DataType, typename BridgeType>
void ParallelizedBridge<DataType, BridgeType>::backward() {

  report_backward_updateweight_last_transfer.reset();

  // Update the status that whether we should calculat the output gradient
  // in the backward loop.
  for (size_t ib = 0; ib < _data_cubes_lower.size(); ib++) {
    _bridges[ib]->needs_to_calc_backward_grad = needs_to_calc_backward_grad;
  }

  stratum.backward();

  /**
   * The following two aggregation steps uses the simple fact that,
   * not matter what bridges we are using, if we are parallelizing with
   * batches, then the aggreation of gradients is always the SUM.
   * This is not the property of each layer, this is the property of the
   * derivitive of multivariant functions.
  **/

  if (p_model_cube != NULL) {
    // After backward, it is the responsibility of ParallelizedConvolutionBridge to merge
    // result back.
    // TODO: each bridge can hold their gradient, in this way, we can save the first for
    // loop. But I do not really so how this could be a bottleneck...
    const size_t n_element = p_model_cube->n_elements;
    DataType * const p_model_data = p_model_cube->p_data;
    const size_t n_partition = _data_cubes_lower.size();
    for (size_t i=0;i<n_element;i++) {
      p_model_data[i] = (-p_model_data[i]) * (n_partition - 1);
    }
    for (size_t i = 0; i < n_partition; ++i) {
      DataType * const p_submodel_data = _bridges[i]->get_model_cube()->p_data;
      for (size_t j=0;j<n_element;j++) {
        p_model_data[j] += p_submodel_data[j];
      }
    }
  }

  if (p_bias_cube != NULL) {
    // do similar things for bias term... Might be better to
    // refactor this to be in the same function as the previous one
    const size_t bias_n_element = p_bias_cube->n_elements;
    DataType * const p_bias_data = p_bias_cube->p_data;
    for (size_t i=0;i<bias_n_element;i++) {
      p_bias_data[i] = (-p_bias_data[i]) * (n_partition - 1);
    }
    for (size_t i = 0; i < n_partition; ++i) {
      DataType * const p_subbias_data = _bridges[i]->get_bias_cube()->p_data;
      for (size_t j=0;j<bias_n_element;j++) {
        p_bias_data[j] += p_subbias_data[j];
      }
    }
  }
  
  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_last_transfer.aggregate_onlystat(stratum.report_backward_updateweight_last_transfer);
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

#endif

