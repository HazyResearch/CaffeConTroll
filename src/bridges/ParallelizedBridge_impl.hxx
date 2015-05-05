//
//  ParallelizedBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 2/8/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_ParallelizedBridge_impl_hxx
#define moka_ParallelizedBridge_impl_hxx

#include <csignal> // or signal.h if C code

// SHADJIS TODO: ParallelizedBridge should not be templated by a single DriverClass.
// All other derived classes of AbstractBridge have a unique driver, so in this way
// ParallelizedBridge is different and perhaps should inherit from some other class
template<typename DataType, 
         template <typename InputLayerDataType, LayoutType InputLayerLayout,
                   typename OutputLayerDataType, LayoutType OutputLayerLayout,
                   typename DriverClass> class BridgeType>
ParallelizedBridge<DataType, BridgeType>::ParallelizedBridge(Layer<DataType,Layout_CRDB> * const _input_layer,
    Layer<DataType, Layout_CRDB> * const _output_layer, const cnn::LayerParameter * const _layer_param,
    const cnn::SolverParameter * const _solver_param, CPUDriver * const _p_driver, size_t _n_partition,
    size_t _n_cpu_thread_per_partition) : AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>(_input_layer,
      _output_layer, _layer_param, _solver_param, _p_driver), n_partition(_n_partition), n_batch(_input_layer->dB),
    n_cpu_thread_per_partition(_n_cpu_thread_per_partition), n_batch_per_partition_cpu(0),
    n_batch_per_partition_gpu(0),
    model_base_learning_rate(1.0),
    bias_base_learning_rate(1.0),
    model_base_regularization(1.0),
    bias_base_regularization(1.0),
    p_model_cube(NULL),
    p_model_grad(NULL),
    p_model_subgrad(NULL),
    p_bias_grad(NULL),
    p_bias_subgrad(NULL),
    p_bias_cube(NULL),
    scheduler_local_cpudriver(NULL),
    scheduler_gpudriver(NULL),
    p_grad_updater(NULL),
    p_grad_updater_bias(NULL),
    extra_partition(false),
    num_partitions(0),
    num_partitions_CPU(0),
    num_partitions_GPU(0)
{
  // Start reporting
  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
  report_backward_updateweight_constructor.reset();
  report_backward_updateweight_last_transfer.reset();
  report_backward_updateweight_history.reset();

  // Begin parallelized bridge constructor

  // Calculate Scheduler parameters (#partitions, partition size, etc.)
  assert(n_batch == curr_B);
  update_scheduler_partitions_for_curr_B();
  assert(num_partitions > 0);
  assert(n_batch_per_partition_cpu + n_batch_per_partition_gpu > 0);
  
  // Right now, we only partition by data, not model
  // First, partition data on the CPU
  size_t b = 0;
  size_t i = 0;
  for ( ; i < num_partitions_CPU; ++i, b += n_batch_per_partition_cpu) {
    const size_t n_batch_this_partition = (extra_partition && i == num_partitions - 1) ? n_batch % n_partition : n_batch_per_partition_cpu;

    // SHADJIS TODO:
    // Here we make 4 cubes for each sub-bridge: input/output and data/gradient
    // Overall we make 4 * num_partitions cubes.
    // These 4 cubes together define 2 layers, the input and output Layer of this sub-bridge
    // So we pair the 2 lower and the 2 higher to make an entry in _partitioned_layers_lower
    // and _partitioned_layers_higher (a Layer is 2 cubes).
    // Those 2 layers together define the sub-bridge.
    // So, a sub-bridge has 2 layers, and a Layer has 2 cubes, i.e. we need 4 cubes.
    //
    // These 4 are not allocated anywhere, but just get pointers of cubes that we want to
    // pass around. So the way we pass things across layers is with the pointers stored in
    // these cubes. This might need to be fixed if we want to pass across layers while
    // keeping on the device, etc.
    //
    // In addition to their use for passing data around, the sizes of these cubes are also
    // used to initialize the AbstractBridge of the sub-bridges we create. So e.g. iC, iB, ...
    // of the abstract bridge for each sub-bridge of the parallelized bridge are defined by
    // the cube sizes below (currently just the data ones).
    
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
  // Next, partition data on the GPU
  // SHADJIS TODO: For multiple GPUs, this would be a loop
  if (num_partitions_GPU > 0) {
    assert(i < num_partitions); // Must be at least 1 partition left
    size_t n_batch_this_partition = n_batch - b; // Remaining batch goes to GPU
    
    // See comment for CPU to understand what these 4 are
    
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
  
  // Create drivers
  
  scheduler_local_cpudriver = p_driver;//new CPUDriver();
  if (num_partitions_GPU > 0) {
#ifdef _INCLUDE_GPUDRIVER
    // SHADJIS TODO: For multi-gpu, pass a stream # into the constructor and make multiple drivers here
    scheduler_gpudriver = new GPUDriver();
#endif
  }  

  for (size_t ib = 0; ib < num_partitions; ib++) {
    _partitioned_layers_lower.push_back(
        new LayerType(_data_cubes_lower[ib], _grad_cubes_lower[ib])
        );

    _partitioned_layers_higher.push_back(
        new LayerType(_data_cubes_higher[ib], _grad_cubes_higher[ib])
        );
  }
  
  // Create bridges
  for (size_t ib = 0; ib < num_partitions; ib++) {
    // Note this constructor passes in the device pointer, i.e. this
    // bridge will have its internal data (e.g. model_cube) allocated
    // on the device.
    if (ib < num_partitions_CPU) {
        _cpu_bridges.push_back(
            new BridgeType<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>(_partitioned_layers_lower[ib], _partitioned_layers_higher[ib],
              layer_param, solver_param, scheduler_local_cpudriver)
            );
    } else {
#ifdef _INCLUDE_GPUDRIVER
        _gpu_bridges.push_back(
            new BridgeType<DataType, Layout_CRDB, DataType, Layout_CRDB, GPUDriver>(_partitioned_layers_lower[ib], _partitioned_layers_higher[ib],
              layer_param, solver_param, scheduler_gpudriver)
            );
#endif
    }
  }
#ifdef _DO_ASSERT
    assert(_cpu_bridges.size() == num_partitions_CPU);
    assert(_gpu_bridges.size() == num_partitions_GPU);
#endif

  for (size_t ib = 0; ib < _cpu_bridges.size(); ib++) {
    _cpu_bridges[ib]->run_with_n_threads = n_cpu_thread_per_partition;
    stratum.executors.push_back((PhysicalOperator *)_cpu_bridges[ib]);
  }
  for (size_t ib = 0; ib < _gpu_bridges.size(); ib++) {
    // _gpu_bridges[ib]->run_with_n_threads = n_cpu_thread_per_partition; // Does nothing on GPU
    stratum.executors.push_back((PhysicalOperator *)_gpu_bridges[ib]);
  }

  stratum.set_executor_bound(stratum.executors.size());

  // create the master model for this parallel bridge
  assert(_cpu_bridges.size() + _gpu_bridges.size() >= 1); // we need at least one partition.
  
  // get one convbrdige as example
  LogicalCubeType * example_bridge_model_cube = NULL;
  bool example_bridge_bias_term = false;
  LogicalCubeType * example_bridge_bias_cube = NULL;
  if (num_partitions_CPU > 0) {
    example_bridge_model_cube = _cpu_bridges[0]->get_model_cube();
    example_bridge_bias_term  = _cpu_bridges[0]->bias_term;
    example_bridge_bias_cube  = _cpu_bridges[0]->get_bias_cube();
  } else {
    example_bridge_model_cube = _gpu_bridges[0]->get_model_cube();
    example_bridge_bias_term  = _gpu_bridges[0]->bias_term;
    example_bridge_bias_cube  = _gpu_bridges[0]->get_bias_cube();
  }
  // TODO: p_model_cube should be T * __const__
  LogicalCubeType * const example_cube = example_bridge_model_cube;
  if (example_cube != NULL) {
    p_model_cube = new LogicalCubeType(example_cube->R, example_cube->C, example_cube->D, example_cube->B);

    // Currently, the parallelized bridge ("scheduler") has a local copy of the model/bias, and
    // this is currently on the host. However, each individual bridge is on the device (CPU, GPU, in future remote, etc.). 
    // This means that we need to get a copy of the weights from one of those bridges.
    // For this, use a device memory pointer memcpy.
    // SHADJIS TODO: Is the local copy of the model even needed? Currently it's only because the gradient update
    // is done on the CPU.
    // SHADJIS TODO: There may be some inneficiency here, since each bridge is allocating and setting the exact same
    // model, even though e.g. if all bridges are on the GPU, we only need to do that once. And if all are on the CPU,
    // there is no need to allocate any models inside the individual bridges, since we can just keep a pointer to the
    // central copy in parallelized bridge.
    // SHADJIS TODO: Every call to get_device_pointer leaks memory?
    
    // Get the driver class of bridges[0]
    if (num_partitions_CPU > 0) {
        scheduler_local_cpudriver->memcpy(p_model_cube->get_device_pointer(scheduler_local_cpudriver), example_cube->get_device_pointer(scheduler_local_cpudriver));
    } else {
        scheduler_gpudriver->memcpy(p_model_cube->get_device_pointer(scheduler_local_cpudriver), example_cube->get_device_pointer(scheduler_gpudriver));
    }

    // Similarly, the parallelized bridge (scheduler) has its own local gradient cube
    // SHADJIS TODO: Should make it clear what is local and not by the variable name
    p_model_grad = new LogicalCubeType(example_cube->R, example_cube->C, example_cube->D, example_cube->B);
    p_model_subgrad = new LogicalCubeType(example_cube->R, example_cube->C, example_cube->D, example_cube->B);

    if (_layer_param->blobs_lr_size() != 0) {
      model_base_learning_rate = _layer_param->blobs_lr(0);
    }
    if (_layer_param->weight_decay_size() != 0) {
      model_base_regularization = _layer_param->weight_decay(0);
    }
    
    // The parallelized bridge also has a SGDGradientUpdater object which updates the gradients given all the sub-bridges
    // For now, do the updating on the CPU
    p_grad_updater = new SGDGradientUpdater<DataType, CPUDriver>(p_model_cube->n_elements, p_model_cube->get_p_data(),
						      _solver_param, model_base_learning_rate, model_base_regularization, scheduler_local_cpudriver);
  } else {
    p_model_cube = NULL;
    p_model_grad = NULL;
    p_model_subgrad = NULL;
  }

  if (example_bridge_bias_term) {
    LogicalCubeType * const example_bias = example_bridge_bias_cube;
    if (example_bias != NULL) {
      p_bias_cube = new LogicalCubeType(example_bias->R, example_bias->C, example_bias->D, example_bias->B);
      // Like above, use a device memcpy here
      if (num_partitions_CPU > 0) {
          scheduler_local_cpudriver->memcpy(p_bias_cube->get_device_pointer(scheduler_local_cpudriver), example_bias->get_device_pointer(scheduler_local_cpudriver));
      } else {
          scheduler_gpudriver->memcpy(p_bias_cube->get_device_pointer(scheduler_local_cpudriver), example_bias->get_device_pointer(scheduler_gpudriver));
      }
      p_bias_grad = new LogicalCubeType(example_bias->R, example_bias->C, example_bias->D, example_bias->B);
      p_bias_subgrad = new LogicalCubeType(example_bias->R, example_bias->C, example_bias->D, example_bias->B);

      if (_layer_param->blobs_lr_size() >1) {
        bias_base_learning_rate = _layer_param->blobs_lr(1);
      }
      if (_layer_param->weight_decay_size() >1) {
        bias_base_regularization = _layer_param->weight_decay(1);
      }

      p_grad_updater_bias = new SGDGradientUpdater<DataType, CPUDriver>(p_bias_cube->n_elements, p_bias_cube->get_p_data(),
							     _solver_param, bias_base_learning_rate, bias_base_regularization, scheduler_local_cpudriver);
    } else {
      p_bias_cube = NULL;
    }
  } else {
    p_bias_cube = NULL;
  }

  // SHADJIS TODO: This constructor takes a couple of seconds when using GPU
  // This is minor for now but may become an issue with many other types of devices
  report_backward_updateweight_constructor.end(0, 0, 0);
  report_forward_constructor.end(0, 0, 0);
}

template<typename DataType, 
         template <typename InputLayerDataType, LayoutType InputLayerLayout,
                   typename OutputLayerDataType, LayoutType OutputLayerLayout,
                   typename DriverClass> class BridgeType>
void ParallelizedBridge<DataType, BridgeType>::forward() {
  report_forward_last_transfer.reset();
  assert(num_partitions <= _cpu_bridges.size() + _gpu_bridges.size());

  // CPU batches
  size_t b = 0;
  size_t i = 0;
  for ( ; i < num_partitions_CPU; ++i, b += n_batch_per_partition_cpu) {
    const size_t n_batch_this_partition = (extra_partition && i == num_partitions - 1) ? curr_B % n_partition : n_batch_per_partition_cpu;
    _data_cubes_lower[i]->set_p_data(p_input_layer->p_data_cube->physical_get_RCDslice(b));
    // Check if this bridge has a model (e.g. not for max pool)
    if (p_model_cube)
    {
        // Special-case for CPU: avoid a copy since device is CPU
        _cpu_bridges[i]->set_model_cube(p_model_cube);
        _cpu_bridges[i]->set_bias_cube(p_bias_cube);
    }
    _cpu_bridges[i]->set_curr_batch_size(n_batch_this_partition);
  }
  // Final GPU batch
  // SHADJIS TODO: For multiple GPUs, this would be a loop
  if (num_partitions_GPU > 0) {
    assert(i < num_partitions); // Must be at least 1 partition left
    size_t n_batch_this_partition = curr_B - b; // Remaining batch goes to GPU
    _data_cubes_lower[i]->set_p_data(p_input_layer->p_data_cube->physical_get_RCDslice(b));
    // Check if this bridge has a model (e.g. not for max pool)
    if (p_model_cube)
    {
        // General-case: copy from host to device
        // SHADJIS TODO: Can share a single pointer for all GPU bridges on the same device (no need to copy to all bridges)
        assert(_gpu_bridges.size() == 1);
        // SHADJIS TODO: Now gpu bridges is size 1, this will increase for multi-GPU
        // (each gpu driver has a device ID and a stream number)
        assert(num_partitions_GPU == 1);
        scheduler_gpudriver->memcpy(_gpu_bridges[0]->get_model_cube()->get_device_pointer(scheduler_gpudriver), p_model_cube->get_device_pointer(scheduler_local_cpudriver));
        scheduler_gpudriver->memcpy(_gpu_bridges[0]->get_bias_cube() ->get_device_pointer(scheduler_gpudriver), p_bias_cube ->get_device_pointer(scheduler_local_cpudriver));
    }
    _gpu_bridges[0]->set_curr_batch_size(n_batch_this_partition);
  }

  // PhysicalStratum also bounded by the current batch size
  stratum.set_executor_bound(num_partitions);
  stratum.forward();

  report_forward_last_transfer.end();
  report_forward_last_transfer.aggregate_onlystat(stratum.report_forward_last_transfer);
  report_forward_history.aggregate(report_forward_last_transfer);
}

template<typename DataType, 
         template <typename InputLayerDataType, LayoutType InputLayerLayout,
                   typename OutputLayerDataType, LayoutType OutputLayerLayout,
                   typename DriverClass> class BridgeType>
void ParallelizedBridge<DataType, BridgeType>::backward() {
  report_backward_updateweight_last_transfer.reset();
  assert(num_partitions <= _cpu_bridges.size() + _gpu_bridges.size());

  // Update the status of whether we should calculate the output gradient
  // in the backward loop.
  for (size_t ib = 0; ib < _cpu_bridges.size(); ib++) {
    _cpu_bridges[ib]->needs_to_calc_backward_grad = needs_to_calc_backward_grad;
  }
  for (size_t ib = 0; ib < _gpu_bridges.size(); ib++) {
    _gpu_bridges[ib]->needs_to_calc_backward_grad = needs_to_calc_backward_grad;
  }

  stratum.set_executor_bound(num_partitions);
  stratum.backward();
  
  /**
   * The following two aggregation steps uses the simple fact that,
   * no matter what bridges we are using, if we are parallelizing with
   * batches, then the aggreation of gradients is always the SUM.
   * This is not the property of each layer, this is the property of the
   * derivitive of multivariant functions.
  **/

  if (p_model_cube != NULL) {
    // After backward, it is the responsibility of ParallelizedBridge to merge
    // result back.
    
    // For each partition, copy the computed gradient back to the parallelized bridge
    // (on the host). 
    // Just like in forward(), if the driver is a CPU driver there is no need to do 
    // any memcpy here (it's all the same pointer).
    if (num_partitions != 1) {
    
      // Iterate over each sub-bridge (partition) and sum the gradients
      p_model_grad->reset_cube(DataType(0.0));
      DataType * const p_grad_data = p_model_grad->get_p_data();
      const size_t n_element = p_model_grad->n_elements;
      for (size_t i = 0; i < num_partitions; ++i) {
        // Store the gradient from each partition in p_model_subgrad
        // If that partition's bridge was on the CPU already, no need for this copy
        if (i < num_partitions_CPU) {
          DataType * const p_subgrad_data = _cpu_bridges[i]->get_model_grad_cube()->get_p_data();
          for (size_t j=0;j<n_element;j++) {
            p_grad_data[j] += p_subgrad_data[j];
          }
        } else {
          scheduler_gpudriver->memcpy(p_model_subgrad->get_device_pointer(scheduler_local_cpudriver), _gpu_bridges[i-_cpu_bridges.size()]->get_model_grad_cube()->get_device_pointer(scheduler_gpudriver));
          DataType * const p_subgrad_data = p_model_subgrad->get_p_data();
          for (size_t j=0;j<n_element;j++) {
            p_grad_data[j] += p_subgrad_data[j];
          }
        }
      }
      
      // Given a gradient, update the model
      
      // SHADJIS TODO:
      // If this is to be done on the device, we need to copy back
      // For now, I will keep gradient updates on the CPU. To go
      // to the GPU instead, add a copy here
      p_grad_updater->update(p_grad_data); // i.e. p_model_grad->get_p_data()
      
    } else {
      // Just 1 partition (sub-bridge)
      // In this special-case, we are not summing results from multiple subgradients
      // (multiple partitions) so there is no need to allocate a temporary buffer.
      if (num_partitions_CPU > 0) {
        p_grad_updater->update(_cpu_bridges[0]->get_model_grad_cube()->get_p_data());
      } else {
        scheduler_gpudriver->memcpy(p_model_subgrad->get_device_pointer(scheduler_local_cpudriver), _gpu_bridges[0]->get_model_grad_cube()->get_device_pointer(scheduler_gpudriver));
      
        // SHADJIS TODO:
        // If this is to be done on the device, we need to copy back
        // For now, I will keep gradient updates on the CPU. To go
        // to the GPU instead, remove the copy above and do the
        // gradient too on the device.
        p_grad_updater->update(p_model_subgrad->get_p_data());
      }

    }

  }
  
  // Repeat for bias cube

  if (p_bias_cube != NULL) {
    // do similar things for bias term... Might be better to
    // refactor this to be in the same function as the previous one
    if (num_partitions > 1) {
      p_bias_grad->reset_cube(DataType(0.0));
      DataType * const p_grad_data = p_bias_grad->get_p_data();
      const size_t bias_n_element = p_bias_grad->n_elements;
      for (size_t i = 0; i < num_partitions; ++i) {
        // If that partition's bridge was on the CPU already, no need for this copy
        if (i < num_partitions_CPU) {
          DataType * const p_subbias_data = _cpu_bridges[i]->get_bias_grad_cube()->get_p_data();
          for (size_t j=0;j<bias_n_element;j++) {
            p_grad_data[j] += p_subbias_data[j];
          }
        } else {
          scheduler_gpudriver->memcpy(p_bias_subgrad->get_device_pointer(scheduler_local_cpudriver), _gpu_bridges[i-_cpu_bridges.size()]->get_bias_grad_cube()->get_device_pointer(scheduler_gpudriver));
          DataType * const p_subbias_data = p_bias_subgrad->get_p_data();
          for (size_t j=0;j<bias_n_element;j++) {
            p_grad_data[j] += p_subbias_data[j];
          }
        }
      }
      
      // SHADJIS TODO:
      // If this is to be done on the device, we need to copy back
      // For now, I will keep gradient updates on the CPU. To go
      // to the GPU instead, add a copy here.
      p_grad_updater_bias->update(p_grad_data);
    } else {
      if (num_partitions_CPU > 0) {
        p_grad_updater_bias->update(_cpu_bridges[0]->get_bias_grad_cube()->get_p_data());
      } else {
        scheduler_gpudriver->memcpy(p_bias_subgrad->get_device_pointer(scheduler_local_cpudriver), _gpu_bridges[0]->get_bias_grad_cube()->get_device_pointer(scheduler_gpudriver));      
        // SHADJIS TODO:
        // If this is to be done on the device, we need to copy back
        // For now, I will keep gradient updates on the CPU. To go
        // to the GPU instead, remove the copy above and do the
        // gradient too on the device.
        p_grad_updater_bias->update(p_bias_subgrad->get_p_data());
      }
    }
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_last_transfer.aggregate_onlystat(stratum.report_backward_updateweight_last_transfer);
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template<typename DataType, 
         template <typename InputLayerDataType, LayoutType InputLayerLayout,
                   typename OutputLayerDataType, LayoutType OutputLayerLayout,
                   typename DriverClass> class BridgeType>
ParallelizedBridge<DataType, BridgeType>::~ParallelizedBridge() {

  for (auto layer = _partitioned_layers_lower.begin(); layer != _partitioned_layers_lower.end(); ++layer) {
    delete (*layer);
  }

  for (auto layer = _partitioned_layers_higher.begin(); layer != _partitioned_layers_higher.end(); ++layer) {
    delete (*layer);
  }

  for (auto bridge = _cpu_bridges.begin(); bridge != _cpu_bridges.end(); ++bridge) {
    delete (*bridge);
  }
  for (auto bridge = _gpu_bridges.begin(); bridge != _gpu_bridges.end(); ++bridge) {
    delete (*bridge);
  }

  if (p_model_cube) {
    delete p_model_cube;
    delete p_model_grad;
    delete p_model_subgrad;
    delete p_grad_updater;
  }

  if (p_bias_cube) {
    delete p_bias_cube;
    delete p_bias_grad;
    delete p_bias_subgrad;
    delete p_grad_updater_bias;
  }
  
  // SHADJIS TODO: Delete this properly (causes warnings)
  // delete scheduler_local_cpudriver; // Only if we don't use p_driver
  // if (scheduler_gpudriver) { delete scheduler_gpudriver; }

}

#endif
