//
//  ParallelizedBridge_impl.hxx
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_ParallelizedBridge_impl_hxx
#define moka_ParallelizedBridge_impl_hxx

#include <csignal>

// SHADJIS TODO: Once we have many GPUs running we may need to do a synchronize 
// at the end of pbridge before going to the next bridge. Currently this is done
// automatically because we copy back from device to host at the end of bridges
// executing on the GPU.

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
    size_t _n_cpu_thread_per_partition,
    // SHADJIS TODO: We now also need the following information from the previous bridge:
    //    - The batch sizes for all devices (CPU, GPUs)
    //    - The pointers to the data cubes for all devices
    // One way to get this data is to just pass in the previous bridge e.g. as a reference.
    // But the type of the bridge (although independent to the above data) is unknown, e.g. 
    // whether it is a parallelized convbridge, fullyconnected bridge, etc., so I have to 
    // template the argument. Instead for now I will just pass in the required information 
    // from deepnet.h. This is also moving towards a global scheduler (i.e. the data passed
    // from deepnet would eventually be passed from a scheduler).
    // SHADJIS TODO: I think the minimum info needed to be sure the partitions are identical
    // are the gpu batch sizes and the gpu to device id map (since we know batch size is same),
    // and then also need num_partitions_CPU (so we know the cube vector indices match, e.g.
    // for batch size 256, bridge 1 could have 8 CPU bridges of size 16 and 1 GPU bridge of size
    // 128, and bridge 2 could have 1 CPU bridge of size 128 and 1 GPU bridge of size 128, as in
    // the case of fully-connected. We can still save a copy even though the size of the vectors
    // previous bridge data cubes higher and grad cubes higher will differ (9 vs 2). (Another
    // solution rather than passing in previous bridge num partitions CPU is just keeping a 
    // vector of the GPU cubes rather than putting CPU + GPU cubes in the same vector)
    const size_t PREVIOUS_BRIDGE_num_partitions_CPU,
    const std::vector<size_t>& PREVIOUS_BRIDGE_GPU_batch_sizes,
    const std::vector<int>   & PREVIOUS_BRIDGE_used_gpu_to_device_id_map,
    const std::vector<LogicalCubeType *> & PREVIOUS_BRIDGE_data_cubes_higher,
    const std::vector<LogicalCubeType *> & PREVIOUS_BRIDGE_grad_cubes_higher,
    // Is the bridge's top and bottom layer the same (i.e. the bridge shares an input and output
    // layer, like ReLU and dropout)
    bool _share_input_output_layer
    // End of class members from previous bridge
    ) : AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>(_input_layer,
      _output_layer, _layer_param, _solver_param, _p_driver),
    p_model_cube(NULL),
    p_model_grad(NULL),
    p_bias_grad(NULL),
    p_bias_subgrad(NULL),
    p_bias_cube(NULL),
    n_partition(_n_partition), // SHADJIS TODO: Maybe should not have this as an argument, but just detect # cores, or in a config file
    n_batch(_input_layer->dB),
    n_cpu_thread_per_partition(_n_cpu_thread_per_partition),
    extra_partition(false),
    num_partitions_CPU(0),
    num_partitions_GPU(0),
    num_partitions(0),
    n_batch_per_partition_cpu(0),
    scheduler_local_cpudriver(NULL),
    share_pointer_with_prev_bridge(false),
    share_pointer_with_next_bridge(false),
    share_input_output_layer(_share_input_output_layer),
    skip_model_copy_gpu(false),
    model_parallelism_group_size(1),
    model_base_learning_rate(1.0),
    bias_base_learning_rate(1.0),
    model_base_regularization(1.0),
    bias_base_regularization(1.0),
    update_model_gradients(true),
    pointer_to_host_copy_of_latest_model_grad(NULL),
    pointer_to_host_copy_of_latest_bias_grad(NULL),
    p_grad_updater(NULL),
    p_grad_updater_bias(NULL)
#ifdef _INCLUDE_GPUDRIVER
    ,gpu_grad_updater(NULL),
    gpu_grad_updater_bias(NULL)
#endif
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
  
  // Now that we know whether we'll be using any GPUs and the batch sizes if
  // we are, we can check if the previous bridge has the same device assignments
  // for the batch elements
  // (This just checks that the pairs of vectors match)
  bool prev_bridge_has_same_device_assignments = do_bridges_have_same_device_assignments(
        GPU_batch_sizes,
        used_gpu_to_device_id_map,
        PREVIOUS_BRIDGE_GPU_batch_sizes,
        PREVIOUS_BRIDGE_used_gpu_to_device_id_map
    );
  // The decision to share data pointers on the device with the previous pbridge 
  share_pointer_with_prev_bridge = prev_bridge_has_same_device_assignments;
  skip_model_copy_gpu = (num_partitions == 1 && num_partitions_GPU == 1);
  // By default every bridge calculates the backward data gradient, except the
  // first conv layer.
  // SHADJIS TODO: More generally, we could skip bw data gradient computation
  // for the first layer which is either conv or fc, as well as everything before it.
  // Also if LR = 0 there is no need to calc the model grad (but that is another
  // bool, currently if there is a model, we always calculate its gradient)
  needs_to_calc_backward_grad = true;
  //needs_to_calc_backward_model_grad = true;

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
    
    // Also, note that here each of these cubes points to a single location of the input
    // and ouput layers, but if batch sizes change (i.e. a bridge can support any batch
    // size) then instead of setting the cube pointers here to point to the input/output
    // layers, instead set the pointers NULL here and set the pointers during the call
    // to forward (for all 4 cubes).
    
    // Also note lower and higher are bad names, they should be input and output
    // because lower is confusing with conv lowering.
    
    _data_cubes_lower.push_back(
        new LogicalCubeType(p_input_layer->p_data_cube->physical_get_RCDslice(b),
          p_input_layer->dR, p_input_layer->dC, p_input_layer->dD, n_batch_this_partition)
        );

    _grad_cubes_lower.push_back(
        new LogicalCubeType(p_input_layer->p_gradient_cube->physical_get_RCDslice(b),
          p_input_layer->gR, p_input_layer->gC, p_input_layer->gD, n_batch_this_partition)
        );

    // SHADJIS TODO: For the GPU, if the bridge has the same top and bottom layer
    // (e.g. this is true for ReLU, dropout) then we re-use the input pointers as 
    // output pointers for each sub-bridge. We do not do this for CPU (below) since
    // deepnet allocated the output cubes already and will use those as the input
    // cubes for the next layer. If in the future we want to save that allocation 
    // on the CPU (i.e. if we want to not make new output cubes for ReLU/dropout in
    // deepnet that own their own data but rather make cubes that point to the input
    // cubes of those bridges), then here we need to set the output cube pointers to 
    // the input cube pointers as we do in the GPU case.
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
  // Before we do that, we need to create GPU drivers
  
  // Create drivers  
  scheduler_local_cpudriver = p_driver;//new CPUDriver();
  // GPU drivers
  for (size_t gpu_i = 0; gpu_i < num_partitions_GPU; ++gpu_i)
  {
#ifdef _INCLUDE_GPUDRIVER
    GPUDriver *new_driver = new GPUDriver();
    new_driver->set_device_id(used_gpu_to_device_id_map[gpu_i]);
    scheduler_gpudrivers.push_back(new_driver); // SHADJIS TODO: Delete these in destructor
#endif
  }
  
  // Now that we've made drivers, create the cubes that will define the sub-bridges on the GPU
  for (size_t gpu_i = 0; gpu_i < num_partitions_GPU; ++gpu_i)
  {
    assert(i < num_partitions); // Must be at least 1 partition left
    size_t n_batch_this_partition = GPU_batch_sizes[gpu_i];
    assert(n_batch_this_partition > 0);
    
    // See comment for CPU to understand what these 4 are
    // Unlike the CPU however, these are allocated on the GPU
    
    // If we are going to just share pointers to device data from the previous
    // bridge, don't own our own input cubes.
    if (share_pointer_with_prev_bridge) {
        _data_cubes_lower.push_back(
            new LogicalCubeType(PREVIOUS_BRIDGE_data_cubes_higher[PREVIOUS_BRIDGE_num_partitions_CPU + gpu_i]->get_p_data(),
              p_input_layer->dR, p_input_layer->dC, p_input_layer->dD, n_batch_this_partition,
              scheduler_gpudrivers[gpu_i])
            );
        _grad_cubes_lower.push_back(
            new LogicalCubeType(PREVIOUS_BRIDGE_grad_cubes_higher[PREVIOUS_BRIDGE_num_partitions_CPU + gpu_i]->get_p_data(),
              p_input_layer->gR, p_input_layer->gC, p_input_layer->gD, n_batch_this_partition,
              scheduler_gpudrivers[gpu_i])
            );
    }
    // Own our own input cubes.
    else {
        _data_cubes_lower.push_back(
            new LogicalCubeType(p_input_layer->dR, p_input_layer->dC, p_input_layer->dD, n_batch_this_partition,
              scheduler_gpudrivers[gpu_i])
            );
        // SHADJIS TODO: This one doesn't need to be allocated if this is the first bridge
        // or if we are running in test mode.
        _grad_cubes_lower.push_back(
            new LogicalCubeType(p_input_layer->gR, p_input_layer->gC, p_input_layer->gD, n_batch_this_partition,
              scheduler_gpudrivers[gpu_i])
            );
    }
 
    // If the top and bottom layers are different, bridges own their output cubes (they have to exist somewhere 
    // on device, so use the convention of "the bridge always owns its output cubes on the device")
    // If the bridge has the same top and bottom layer (e.g. this is true for ReLU, dropout)
    // then re-use the input cube pointers as output cube pointers for each sub-bridge (no allocation)
    if (share_input_output_layer) {
        _data_cubes_higher.push_back(
            new LogicalCubeType(_data_cubes_lower.back()->get_p_data(),
              p_input_layer->dR, p_input_layer->dC, p_input_layer->dD, n_batch_this_partition,
              scheduler_gpudrivers[gpu_i])
            );
        _grad_cubes_higher.push_back(
            new LogicalCubeType(_grad_cubes_lower.back()->get_p_data(),
              p_input_layer->gR, p_input_layer->gC, p_input_layer->gD, n_batch_this_partition,
              scheduler_gpudrivers[gpu_i])
            );
    } else {
        _data_cubes_higher.push_back(
            new LogicalCubeType(p_output_layer->dR, p_output_layer->dC, p_output_layer->dD, n_batch_this_partition,
              scheduler_gpudrivers[gpu_i])
            );
        _grad_cubes_higher.push_back(
            new LogicalCubeType(p_output_layer->gR, p_output_layer->gC, p_output_layer->gD, n_batch_this_partition,
              scheduler_gpudrivers[gpu_i])
            );
    }
        
    b += n_batch_this_partition;
    ++i;
  }
  if (num_partitions_GPU > 0) {
    assert(b == n_batch); // Must have completed every image now
  }
  assert(_data_cubes_lower.size() == num_partitions); // 4 cubes per partition (since 4 cubes per bridge)

  // Now we've made all our cubes, so we can make layers
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
              layer_param, solver_param, scheduler_gpudrivers[ib-_cpu_bridges.size()])
            );
#endif
    }
  }

  assert(_cpu_bridges.size() == num_partitions_CPU);
  assert(_gpu_bridges.size() == num_partitions_GPU);
  assert(_gpu_bridges.size() == scheduler_gpudrivers.size());

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

  // get one bridge as example
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
    
    // First check for the blobs_lr and weight_decay
    // These are old and can be removed in future versions
    if (_layer_param->blobs_lr_size() != 0) {
      model_base_learning_rate = _layer_param->blobs_lr(0);
    }
    if (_layer_param->weight_decay_size() != 0) {
      model_base_regularization = _layer_param->weight_decay(0);
    }
    // If not, we are using the new format,
    //
    //  param {
    //    lr_mult: 1
    //    decay_mult: 1
    //  }
    //
    if (_layer_param->param_size() != 0) {
      model_base_learning_rate  = _layer_param->param(0).lr_mult();
      model_base_regularization = _layer_param->param(0).decay_mult();
    }
    
    
    // If we have a single GPU partition, updates will be on that GPU
    // There is no need then to allocate any model on the CPU
    // SHADJIS TODO: Support gradient updates on all GPUs without having to go through host
    if (skip_model_copy_gpu) {
#ifdef _INCLUDE_GPUDRIVER
        // This p_model_cube represents the CPU's local copy of the model cube
        // Since the model is entirely on the GPU this may not be needed, however
        // if we ever want to read or write the model from/to a file we will need
        // a host copy, so that will be stored here.
        p_model_cube = new LogicalCubeType(example_cube->R, example_cube->C, example_cube->D, example_cube->B);
        // Similarly p_model_grad isn't needed anymore but is a buffer we can keep 
        // in case we want a host copy, e.g. for returning the gradients to some
        // parameter server
        p_model_grad = new LogicalCubeType(example_cube->R, example_cube->C, example_cube->D, example_cube->B);
        gpu_grad_updater = new SGDGradientUpdater<DataType, GPUDriver>(example_cube->n_elements, example_cube->get_p_data(),
                                  _solver_param, model_base_learning_rate, model_base_regularization, scheduler_gpudrivers[0]);
#endif
    } else {
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
        // SHADJIS TODO: This copy on CPU isn't needed, can just set pointer
        if (num_partitions_CPU > 0) {
            scheduler_local_cpudriver->memcpy(p_model_cube->get_device_pointer(scheduler_local_cpudriver), example_cube->get_device_pointer(scheduler_local_cpudriver));
        } else {
            // We can pick any GPU to copy from, so let's pick 0 since
            // it is always the fastest one
            scheduler_gpudrivers[0]->memcpy(p_model_cube->get_device_pointer(scheduler_local_cpudriver), example_cube->get_device_pointer(scheduler_gpudrivers[0]));
        }

        // Similarly, the parallelized bridge (scheduler) has its own local gradient cube
        // SHADJIS TODO: Should make it clear what is local and not by the variable name
        p_model_grad = new LogicalCubeType(example_cube->R, example_cube->C, example_cube->D, example_cube->B); // This isn't needed anymore
        for (size_t pi=0; pi<num_partitions_GPU; ++pi) {
          p_model_subgrads.push_back(new LogicalCubeType(example_cube->R, example_cube->C, example_cube->D, example_cube->B));
        }

        // The parallelized bridge also has a SGDGradientUpdater object which updates the gradients given all the sub-bridges
        // For now, do the updating on the CPU
        p_grad_updater = new SGDGradientUpdater<DataType, CPUDriver>(p_model_cube->n_elements, p_model_cube->get_p_data(),
                                  _solver_param, model_base_learning_rate, model_base_regularization, scheduler_local_cpudriver);
    }
    
  } else {
    p_model_cube = NULL;
    p_model_grad = NULL;
  }

  if (example_bridge_bias_term) {
    LogicalCubeType * const example_bias = example_bridge_bias_cube;
    if (example_bias != NULL) {


      // First check for the blobs_lr and weight_decay
      // These are old and can be removed in future versions
      if (_layer_param->blobs_lr_size() >1) {
        bias_base_learning_rate = _layer_param->blobs_lr(1);
      }
      if (_layer_param->weight_decay_size() >1) {
        bias_base_regularization = _layer_param->weight_decay(1);
      }
      // If not, we are using the new format,
      //
      //  param {
      //    lr_mult: 1
      //    decay_mult: 1
      //  }
      //
      if (_layer_param->param_size() > 1) {
        bias_base_learning_rate  = _layer_param->param(1).lr_mult();
        bias_base_regularization = _layer_param->param(1).decay_mult();
      }
        
      // If we have a single GPU partition, updates will be on that GPU
      // There is no need then to allocate any model on the CPU
      // SHADJIS TODO: Support gradient updates on all GPUs without having to go through host
      if (skip_model_copy_gpu) {
#ifdef _INCLUDE_GPUDRIVER
        // This p_bias_cube represents the CPU's local copy of the bias cube
        // Since the bias is entirely on the GPU this may not be needed, however
        // if we ever want to read or write the bias from/to a file we will need
        // a host copy, so that will be stored here.
        p_bias_cube = new LogicalCubeType(example_bias->R, example_bias->C, example_bias->D, example_bias->B);
        // Same as comment above for p_model_grad, might need e.g. for a parameter server
        p_bias_grad = new LogicalCubeType(example_bias->R, example_bias->C, example_bias->D, example_bias->B);
        gpu_grad_updater_bias = new SGDGradientUpdater<DataType, GPUDriver>(example_bias->n_elements, example_bias->get_p_data(),
                                _solver_param, bias_base_learning_rate, bias_base_regularization, scheduler_gpudrivers[0]);
#endif
      } else {
        p_bias_cube = new LogicalCubeType(example_bias->R, example_bias->C, example_bias->D, example_bias->B);
        // Like above, use a device memcpy here
        if (num_partitions_CPU > 0) {
            scheduler_local_cpudriver->memcpy(p_bias_cube->get_device_pointer(scheduler_local_cpudriver), example_bias->get_device_pointer(scheduler_local_cpudriver));
        } else {
            scheduler_gpudrivers[0]->memcpy(p_bias_cube->get_device_pointer(scheduler_local_cpudriver), example_bias->get_device_pointer(scheduler_gpudrivers[0]));
        }
        p_bias_grad = new LogicalCubeType(example_bias->R, example_bias->C, example_bias->D, example_bias->B);
        p_bias_subgrad = new LogicalCubeType(example_bias->R, example_bias->C, example_bias->D, example_bias->B);   // Used when #partitions > 1 and #GPU partitions > 0
        
        p_grad_updater_bias = new SGDGradientUpdater<DataType, CPUDriver>(p_bias_cube->n_elements, p_bias_cube->get_p_data(),
                                _solver_param, bias_base_learning_rate, bias_base_regularization, scheduler_local_cpudriver);
      }
    } else {
      p_bias_cube = NULL;
    }
  } else {
    p_bias_cube = NULL;
  }
  
  if (share_pointer_with_prev_bridge) {
    std::cout << "    Sharing data pointers with previous bridge (no input cube allocation or data copy)\n";
  }
  if (p_model_cube || p_bias_cube) {
      if (skip_model_copy_gpu) {
        std::cout << "    Gradient updates will happen on the device\n";
      } else {
        std::cout << "    Gradient updates will happen on the host\n";
      }
  } 
  if (share_input_output_layer) {
    std::cout << "    Input and output layer cubes will share data (no output cube allocation)\n";
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
  // Assert less than or equal since we might have a smaller final batch
#ifdef _DO_ASSERT
  assert(num_partitions == _cpu_bridges.size() + _gpu_bridges.size()); // SHADJIS TODO: For variable batch size, assert <=
#endif

  // Iterate over all bridges and set the model cube for this iteration
  PROFILE_ONLY(Timer t; float seconds;)
  
  // CPU batches
  for (size_t i = 0; i < num_partitions_CPU; ++i) {
    // Check if this bridge has a model (e.g. not for max pool)
    if (p_model_cube)
    {
        // Special-case for CPU: avoid a copy since device is CPU
        // SHADJIS TODO: Do we still allocate these cubes? No need if we
        // just share the pointer
        _cpu_bridges[i]->set_model_cube(p_model_cube);
        _cpu_bridges[i]->set_bias_cube(p_bias_cube);
    }
  }
  // GPU batches
  // Recall we might not want to copy to the GPU if it updates the
  // gradients itself
  // Also Check if this bridge has a model (e.g. not for max pool)
  // SHADJIS TODO: code below assumes p_bias_cube exists because p_model_cube does
  if (!skip_model_copy_gpu && p_model_cube)
  {
    // We will use threads to do async copies to multiple GPUs
    // SHADJIS TODO: Can use cuda memcpy async but need to pin memory?
    if (num_partitions_GPU == 1) { // Special case: Don't launch threads
        scheduler_gpudrivers[0]->memcpy(_gpu_bridges[0]->get_model_cube()->get_device_pointer(scheduler_gpudrivers[0]), p_model_cube->get_device_pointer(scheduler_local_cpudriver));
        scheduler_gpudrivers[0]->memcpy(_gpu_bridges[0]->get_bias_cube() ->get_device_pointer(scheduler_gpudrivers[0]), p_bias_cube ->get_device_pointer(scheduler_local_cpudriver));
    } else if (num_partitions_GPU > 1) {
      vector<thread> threads;
      for (size_t gpu_i = 0; gpu_i < num_partitions_GPU; ++gpu_i) {
        threads.push_back(thread([this, gpu_i]() {
            // General-case: copy from host to device
            scheduler_gpudrivers[gpu_i]->memcpy(_gpu_bridges[gpu_i]->get_model_cube()->get_device_pointer(scheduler_gpudrivers[gpu_i]), p_model_cube->get_device_pointer(scheduler_local_cpudriver));
            scheduler_gpudrivers[gpu_i]->memcpy(_gpu_bridges[gpu_i]->get_bias_cube() ->get_device_pointer(scheduler_gpudrivers[gpu_i]), p_bias_cube ->get_device_pointer(scheduler_local_cpudriver));
        }));
      }
      for (size_t ti = 0; ti < threads.size(); ti++) {
        threads[ti].join();
      }
    }
  }
  PROFILE_ONLY(seconds = t.elapsed(); std::cout << "    PB:  Copy Model -> Device:    " << seconds << "\n"; t.restart(); )

  // Also, iterate over all bridges and set the data cubes
  // These were set in the constructor. Recall the data cubes defining each sub-bridge are just
  // pointers to the right places (RCD slices) of the input and output layer of the PBridge
  // There are 2 exceptions:
  //  1. For variable batch size (not supported now or maybe ever), might need to update that
  //     RCD slice pointer (since the batch size is different so batch 2 starts at a different
  //     place in memory than it did before), and
  //  2. For GPU, we have to update the data since it may need to be copied to the device again
  //     (it may not be, if the previous bridge also had the same data on that GPU)
  
  // SHADJIS TODO: Currently I only implement 2 above. If in the future we want to handle 
  // variable batch sizes, then uncomment the CPU loop below and uncomment the GPU portion which 
  // sets the batch sizes and updates the pointers to the input/output layers (currently commented 
  // out but shown for the data cube. Needs to be done like that for all 4 cubes). If that's not 
  // ever needed (since we wrap around dataset during training), then just delete these comments 
  // and also set_curr_batch_size(), and update_scheduler_partitions_for_curr_B(), etc.

  // CPU batches
//  size_t b = 0; // keep track of batch index
//  size_t i = 0; // Move i outside of 1st loop
//  for ( ; i < num_partitions_CPU; ++i, b += n_batch_per_partition_cpu) {
//    const size_t n_batch_this_partition = (extra_partition && i == num_partitions - 1) ? curr_B % n_partition : n_batch_per_partition_cpu;
//    _data_cubes_lower[i]->set_p_data(p_input_layer->p_data_cube->physical_get_RCDslice(b));
//    _cpu_bridges[i]->set_curr_batch_size(n_batch_this_partition);
//  }
// And then uncomment the relevant lines in the loop below, i.e. to set the proper
// curr_B and also adjust the location from which we copy from device.

  // Note: We don't always copy from the host to the device. We only do this
  // when the previous bridge did not share device data with the current bridge.
  if (!share_pointer_with_prev_bridge)
  {
      // GPU batches
      if (num_partitions_GPU == 1) { // Special case: Don't launch threads
            // assert(i < num_partitions); // Must be at least 1 partition left
            
            // Now copy to GPU
            // Remember that e.g. _data_cubes_lower defines the p_input_layer for each sub-bridge
            // Inside the bridge, it will assume _data_cubes_lower (called p_input_layer->p_data_cube
            // inside the bridge) is already allocated on the GPU. This is true for all 4 cubes.
            
            // Get the amount to copy to GPU
            size_t n_batch_this_partition = GPU_batch_sizes[0];
    #ifdef _DO_ASSERT
            assert(n_batch_this_partition > 0);
    #endif
            size_t b = num_partitions_CPU * n_batch_per_partition_cpu;

            // Copy from location b of the PBridge's data/gradient layers
            scheduler_gpudrivers[0]->memcpy(
                // dst = the cube in the _data_cubes_lower vector corresponding to this GPU
                // (this is going to be the input data for the GPU)
                _data_cubes_lower[0 + num_partitions_CPU]->get_device_pointer(scheduler_gpudrivers[0]),
                // src = the cpu data in the input layer starting at the right batch (batch "b")
                p_input_layer->p_data_cube->get_device_pointer_RCDslice(scheduler_local_cpudriver, b, n_batch_this_partition)
            );
            
            // Update the pointer to the right batch for the next sub-bridge
            // b += n_batch_this_partition;
            
            // More things only relevant for variable batch size:
            // _gpu_bridges[gpu_i]->set_curr_batch_size(n_batch_this_partition);
            // ++i;
      } else if (num_partitions_GPU > 1) {
          vector<thread> threads;
          for (size_t gpu_i = 0; gpu_i < num_partitions_GPU; ++gpu_i)
          {
            threads.push_back(thread([this, gpu_i]() {
                // assert(i < num_partitions); // Must be at least 1 partition left
                
                // Now copy to GPU
                // Remember that e.g. _data_cubes_lower defines the p_input_layer for each sub-bridge
                // Inside the bridge, it will assume _data_cubes_lower (called p_input_layer->p_data_cube
                // inside the bridge) is already allocated on the GPU. This is true for all 4 cubes.
                
                // Get the amount to copy to GPU
                size_t n_batch_this_partition = GPU_batch_sizes[gpu_i];
        #ifdef _DO_ASSERT
                assert(n_batch_this_partition > 0);
        #endif
                size_t b = num_partitions_CPU * n_batch_per_partition_cpu;
                for (size_t bi=0; bi<gpu_i; ++bi) {
                    b += GPU_batch_sizes[bi];
                }
                
                // Copy from location b of the PBridge's data/gradient layers
                scheduler_gpudrivers[gpu_i]->memcpy(
                    // dst = the cube in the _data_cubes_lower vector corresponding to this GPU
                    // (this is going to be the input data for the GPU)
                    _data_cubes_lower[gpu_i + num_partitions_CPU]->get_device_pointer(scheduler_gpudrivers[gpu_i]),
                    // src = the cpu data in the input layer starting at the right batch (batch "b")
                    p_input_layer->p_data_cube->get_device_pointer_RCDslice(scheduler_local_cpudriver, b, n_batch_this_partition)
                );
                
                // Update the pointer to the right batch for the next sub-bridge
                // b += n_batch_this_partition;
                
                // More things only relevant for variable batch size:
                // _gpu_bridges[gpu_i]->set_curr_batch_size(n_batch_this_partition);
                // ++i;
            }));
          }
          for (size_t ti = 0; ti < threads.size(); ti++) {
            threads[ti].join();
          }
      }
  }
  PROFILE_ONLY(seconds = t.elapsed(); std::cout << "    PB:  Copy Data -> Device:     " << seconds << "\n"; t.restart(); )

  // Now do the actual forward pass for each sub-bridge
  
  // PhysicalStratum also bounded by the current batch size
  stratum.set_executor_bound(num_partitions);
  stratum.forward();

  PROFILE_ONLY(t.restart();)
  
  // Now that the forward step is complete, copy back the data to the host
  // This is analogous to the copy earlier in this function from host -> device
  // This time however, there is no need to copy back to the host if we know the
  // next bridge will also be on the device.
  if (!share_pointer_with_next_bridge)
  {
      // GPU batches
      if (num_partitions_GPU == 1) { // Special case: Don't launch threads
                // Get the amount to copy from GPU
                size_t n_batch_this_partition = GPU_batch_sizes[0];
        #ifdef _DO_ASSERT
                assert(n_batch_this_partition > 0);
        #endif
                size_t b = num_partitions_CPU * n_batch_per_partition_cpu;
                
                // Copy from the GPU's output cube back to the location b of the PBridge's data/gradient layers
                scheduler_gpudrivers[0]->memcpy(
                    // dst = the cpu data in the output layer starting at the right batch (batch "b")
                    p_output_layer->p_data_cube->get_device_pointer_RCDslice(scheduler_local_cpudriver, b, n_batch_this_partition),
                    // src = the cube in the _data_cubes_higher vector corresponding to this GPU
                    // (this is where the output data for the GPU was stored during the bridge execution)
                    _data_cubes_higher[0 + num_partitions_CPU]->get_device_pointer(scheduler_gpudrivers[0])
                );
                // Update the pointer to the right batch for the next sub-bridge
                // b += n_batch_this_partition;
      } else if (num_partitions_GPU > 1) {
          vector<thread> threads;
          for (size_t gpu_i = 0; gpu_i < num_partitions_GPU; ++gpu_i)
          {
            threads.push_back(thread([this, gpu_i]() {
                // Get the amount to copy from GPU
                size_t n_batch_this_partition = GPU_batch_sizes[gpu_i];
        #ifdef _DO_ASSERT
                assert(n_batch_this_partition > 0);
        #endif
                size_t b = num_partitions_CPU * n_batch_per_partition_cpu;
                for (size_t bi=0; bi<gpu_i; ++bi) {
                    b += GPU_batch_sizes[bi];
                }
          
                // Copy from the GPU's output cube back to the location b of the PBridge's data/gradient layers
                scheduler_gpudrivers[gpu_i]->memcpy(
                    // dst = the cpu data in the output layer starting at the right batch (batch "b")
                    p_output_layer->p_data_cube->get_device_pointer_RCDslice(scheduler_local_cpudriver, b, n_batch_this_partition),
                    // src = the cube in the _data_cubes_higher vector corresponding to this GPU
                    // (this is where the output data for the GPU was stored during the bridge execution)
                    _data_cubes_higher[gpu_i + num_partitions_CPU]->get_device_pointer(scheduler_gpudrivers[gpu_i])
                );
                // Update the pointer to the right batch for the next sub-bridge
                // b += n_batch_this_partition;
            }));
          }
          for (size_t ti = 0; ti < threads.size(); ti++) {
            threads[ti].join();
          }
      }
  }
  // SHADJIS TODO: It makes sense to sync here so we do not return too early
  // before the computations are done. Syncing is automatically done by copying
  // back to the host but since sometimes we skip this, I'll add a sync here for
  // each device. Might not be necessary though.
  else {
      for (size_t gpu_i = 0; gpu_i < num_partitions_GPU; ++gpu_i) {
        scheduler_gpudrivers[gpu_i]->device_sync();
      }
  }
  PROFILE_ONLY(seconds = t.elapsed(); std::cout << "    PB:  Copy Data -> Host:       " << seconds << "\n"; t.restart(); )

  report_forward_last_transfer.end();
  report_forward_last_transfer.aggregate_onlystat(stratum.report_forward_last_transfer);
  report_forward_history.aggregate(report_forward_last_transfer);
}

template<typename DataType, 
         template <typename InputLayerDataType, LayoutType InputLayerLayout,
                   typename OutputLayerDataType, LayoutType OutputLayerLayout,
                   typename DriverClass> class BridgeType>
// This takes an argument (default true) that specifies whether we should update 
// model gradients during the backward pass
void ParallelizedBridge<DataType, BridgeType>::backward() {
  report_backward_updateweight_last_transfer.reset();
#ifdef _DO_ASSERT
  assert(num_partitions == _cpu_bridges.size() + _gpu_bridges.size()); // SHADJIS TODO: For variable batch size, assert <=
#endif

  PROFILE_ONLY(Timer t; float seconds;)
  
  // Update the status of whether we should calculate the output gradient
  // in the backward loop.
  for (size_t ib = 0; ib < _cpu_bridges.size(); ib++) {
    _cpu_bridges[ib]->needs_to_calc_backward_grad = needs_to_calc_backward_grad;
  }
  for (size_t ib = 0; ib < _gpu_bridges.size(); ib++) {
    _gpu_bridges[ib]->needs_to_calc_backward_grad = needs_to_calc_backward_grad;
  }
  
  // Copy to device, if necessary
  if (!share_pointer_with_next_bridge)
  {
      if (num_partitions_GPU == 1) { // Special case: Don't launch threads
        // Get the amount to copy to GPU
        size_t n_batch_this_partition = GPU_batch_sizes[0];
#ifdef _DO_ASSERT
        assert(n_batch_this_partition > 0);
#endif
        size_t b = num_partitions_CPU * n_batch_per_partition_cpu;
        
        // Copy from location b of the PBridge's data/gradient layers
        scheduler_gpudrivers[0]->memcpy(
            // dst = the cube in the _data_cubes_lower vector corresponding to this GPU
            // (this is going to be the input data for the GPU)
            _grad_cubes_higher[0 + num_partitions_CPU]->get_device_pointer(scheduler_gpudrivers[0]),
            // src = the cpu data in the input layer starting at the right batch (batch "b")
            p_output_layer->p_gradient_cube->get_device_pointer_RCDslice(scheduler_local_cpudriver, b, n_batch_this_partition)
        );
        // Update the pointer to the right batch for the next sub-bridge
        // b += n_batch_this_partition;
      } else if (num_partitions_GPU > 1) {
          vector<thread> threads;
          for (size_t gpu_i = 0; gpu_i < num_partitions_GPU; ++gpu_i)
          {
            threads.push_back(thread([this, gpu_i]() {
                // Get the amount to copy to GPU
                size_t n_batch_this_partition = GPU_batch_sizes[gpu_i];
        #ifdef _DO_ASSERT
                assert(n_batch_this_partition > 0);
        #endif
                size_t b = num_partitions_CPU * n_batch_per_partition_cpu;
                for (size_t bi=0; bi<gpu_i; ++bi) {
                    b += GPU_batch_sizes[bi];
                }
                
                // Copy from location b of the PBridge's data/gradient layers
                scheduler_gpudrivers[gpu_i]->memcpy(
                    // dst = the cube in the _data_cubes_lower vector corresponding to this GPU
                    // (this is going to be the input data for the GPU)
                    _grad_cubes_higher[gpu_i + num_partitions_CPU]->get_device_pointer(scheduler_gpudrivers[gpu_i]),
                    // src = the cpu data in the input layer starting at the right batch (batch "b")
                    p_output_layer->p_gradient_cube->get_device_pointer_RCDslice(scheduler_local_cpudriver, b, n_batch_this_partition)
                );
                // Update the pointer to the right batch for the next sub-bridge
                // b += n_batch_this_partition;
            }));
          }
          for (size_t ti = 0; ti < threads.size(); ti++) {
            threads[ti].join();
          }
      }
  }
  PROFILE_ONLY(seconds = t.elapsed(); std::cout << "    PB:  Copy Data -> Device:     " << seconds << "\n"; t.restart(); )

  stratum.set_executor_bound(num_partitions);
  stratum.backward();

  PROFILE_ONLY(t.restart();)

  /**
   * Aggregate gradients. No matter what bridges we are using, if we are 
   * parallelizing with batches then the aggreation is always the sum.
  **/
  
  // If we are going to be skipping the model copy, then the GPUs
  // will be in charge of their own gradient updates. That means that we 
  // need to initialize cuBLAS for the thread running this pbridge, since
  // the update is run as part of the same thread.
  // SHADJIS TODO: Now I call this for every pbridge backward pass, may
  // be okay to just call once (not for each pbridge and each iteration)
  if (skip_model_copy_gpu) {
    scheduler_gpudrivers[0]->init_thread();
  }

  // SHADJIS TODO: When using multiple GPUs we need to combine gradients for each one.
  // The naive way is to copy all the gradients back to the host and sum them there,
  // then copy the new model back to each GPU. This aggregation can be slow on CPU 
  // however (update is saxpy) and also requires many copies back. We need to think
  // of how to solve the problem for a distributed setting.
  //
  // For now, handle the special-case the case of 1 GPU. In this case, the GPU is 
  // used to calculate its own gradient update and does not need the weights to
  // be copied to it again before the forward pass.
  // 
  // In the similar case of 1 GPU + also some portion of the batch on CPU, again the 
  // GPU can calculate its own gradient, but now also needs to be given the graidents 
  // from the CPU. That case will also be handled later, not in the current version.
  // Then later also generalize to multi-GPU (e.g. in a similar way, each other GPU 
  // can pass gradients to GPU 0, etc.)
  if (p_model_cube != NULL) {
    // After backward, it is the responsibility of ParallelizedBridge to merge
    // result back.
    
    // For each partition, copy the computed gradient back to the parallelized bridge
    // (on the host). 
    // Just like in forward(), if the driver is a CPU driver there is no need to do 
    // any memcpy here (it's all the same pointer).
    if (num_partitions != 1) {
    
      // For multiple partitions, we will do this:
      // - Copy back the gradients in parallel
      // - Do a serial reduction on CPU
      // - Sum final gradient to model
      // Can do e.g. on GPU later
      
      // SHADJIS TODO: The copy from the GPUs takes longer than the 
      // saxpy to accumulate gradients. I launch a separate thread to
      // read from GPUs in parallel but can think how to optimize further.
      
      // First, get all the gradients on the host
      vector<DataType *> gradients_host_pointers;
      gradients_host_pointers.resize(num_partitions);
      // Iterate over each sub-bridge (partition) and get the gradients
      // If sub-bridge on CPU, just get the pointer
      // Otherwise, do a copy
      vector<thread> threads_for_memcpy;
      for (size_t i = 0; i < num_partitions; ++i) {
        // Store the gradient from each partition in p_model_subgrad
        // If that partition's bridge was on the CPU already, no need for this copy
        if (i < num_partitions_CPU) {
            gradients_host_pointers[i] = _cpu_bridges[i]->get_model_grad_cube()->get_p_data();
        } else {
          threads_for_memcpy.push_back(thread([this, i, &gradients_host_pointers]() { // Capture gradients_host_pointers since not in this object, and capture by ref
              scheduler_gpudrivers[i-num_partitions_CPU]->memcpy(p_model_subgrads[i-num_partitions_CPU]->get_device_pointer(scheduler_local_cpudriver), _gpu_bridges[i-num_partitions_CPU]->get_model_grad_cube()->get_device_pointer(scheduler_gpudrivers[i-num_partitions_CPU]));
              gradients_host_pointers[i] = p_model_subgrads[i-num_partitions_CPU]->get_p_data();
          }));
        }
      }
      // Note, can join later instead
      for (size_t ti = 0; ti < threads_for_memcpy.size(); ti++) {
        threads_for_memcpy[ti].join();
      }

      PROFILE_ONLY(seconds = t.elapsed(); std::cout << "        Update copy:    " << seconds << "\n"; t.restart(); )
  
      // Iterate over each sub-bridge (partition) and get the gradients
      // p_model_grad->reset_cube(DataType(0.0)); // SHADJIS TODO: p_model_grad unused now, can remove
      DataType * const p_grad_data = gradients_host_pointers[0];
      const size_t n_element = p_model_grad->n_elements;

      // SHADJIS TODO: Here I do num_partitions saxpy calls, on CPU. Instead of
      // doing them linearly O(num_partitions) can use a better reduction or do
      // this on GPU
      // Edit: SHADJIS TODO: Minimize #writes by reordering loops. Can handle
      // more special-cases later.
      if (num_partitions == 4) {
        // const DataType * const input0 = gradients_host_pointers[0];
        const DataType * const input1 = gradients_host_pointers[1];
        const DataType * const input2 = gradients_host_pointers[2];
        const DataType * const input3 = gradients_host_pointers[3];
        for (size_t j=0;j<n_element;++j) {
          p_grad_data[j] += ( input1[j] + input2[j] + input3[j] );
          // p_grad_data[j] = ( input0[j] + input1[j] + input2[j] + input3[j] );
        }
      }
      else {
        for (size_t i = 1; i < num_partitions; ++i) {
          // Store the gradient from each partition in p_model_subgrad
          // If that partition's bridge was on the CPU already, no need for this copy
          //if (i < num_partitions_CPU) {
          // // scheduler_local_cpudriver->set_num_threads(n_partition*n_cpu_thread_per_partition);
          // // scheduler_local_cpudriver->math_saxpy(n_element, 1.0, gradients_host_pointers[i], p_grad_data);
          // for (size_t j=0;j<n_element;++j) {
          //   p_grad_data[j] += gradients_host_pointers[i][j];
          // }
          //} else {
          //  // scheduler_local_cpudriver->set_num_threads(n_partition*n_cpu_thread_per_partition);
          //  // scheduler_local_cpudriver->math_saxpy(n_element, 1.0, gradients_host_pointers[i], p_grad_data);
          //  for (size_t j=0;j<n_element;++j) {
          //    p_grad_data[j] += gradients_host_pointers[i][j];
          //  }
          //}
          for (size_t j=0;j<n_element;++j) {
            p_grad_data[j] += gradients_host_pointers[i][j];
          }
        }
      }
      
      PROFILE_ONLY(seconds = t.elapsed(); std::cout << "        Update step 0:  " << seconds << "\n"; t.restart(); )      
  
      // Given a gradient, update the model
      
      // SHADJIS TODO:
      // If this is to be done on the device, we need to copy back
      // For now, I will keep gradient updates on the CPU. To go
      // to the GPU instead, add a copy here
      //scheduler_local_cpudriver->set_num_threads(n_partition*n_cpu_thread_per_partition);
      if (update_model_gradients) {
        p_grad_updater->update(p_grad_data);
      }
      // Save this pointer for later
      pointer_to_host_copy_of_latest_model_grad = p_grad_data;
      
    } else {
    
      // Just 1 partition (sub-bridge)
      
      // In this special-case, we are not summing results from multiple subgradients
      // (multiple partitions) so there is no need to allocate a temporary buffer.
      if (num_partitions_CPU > 0) {
        if (update_model_gradients) {
          p_grad_updater->update(_cpu_bridges[0]->get_model_grad_cube()->get_p_data());
        }
        // Save this pointer for later
        pointer_to_host_copy_of_latest_model_grad = _cpu_bridges[0]->get_model_grad_cube()->get_p_data();
      } else {
        // Here handle the special-case of a single GPU. The GPU can
        // sum its own gradient to its model and therefore we can eliminate a copy
        // of the gradients back to the host, the CPU saxpy updates, and the copy
        // of the new model back to the device.
#ifdef _INCLUDE_GPUDRIVER
    #ifdef _DO_ASSERT
        assert(skip_model_copy_gpu);
    #endif
        if (update_model_gradients) {
          gpu_grad_updater->update(_gpu_bridges[0]->get_model_grad_cube()->get_p_data());
        }
        // Save this pointer for later (NULL since on device)
        pointer_to_host_copy_of_latest_model_grad = NULL;
#endif
      }

    }

  }
  PROFILE_ONLY(seconds = t.elapsed(); std::cout << "    PB:  Update model:            " << seconds << "\n"; t.restart(); )
  
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
          scheduler_gpudrivers[i-num_partitions_CPU]->memcpy(p_bias_subgrad->get_device_pointer(scheduler_local_cpudriver), _gpu_bridges[i-num_partitions_CPU]->get_bias_grad_cube()->get_device_pointer(scheduler_gpudrivers[i-num_partitions_CPU]));
          DataType * const p_subbias_data = p_bias_subgrad->get_p_data();
          for (size_t j=0;j<bias_n_element;++j) {
            p_grad_data[j] += p_subbias_data[j];
          }
        }
      }
      
      // SHADJIS TODO:
      // If this is to be done on the device, we need to copy back
      // For now, I will keep gradient updates on the CPU. To go
      // to the GPU instead, add a copy here.
      if (update_model_gradients) {
        p_grad_updater_bias->update(p_grad_data);
      }
      // Save this pointer for later
      pointer_to_host_copy_of_latest_bias_grad = p_grad_data;
    } else {
    
      if (num_partitions_CPU > 0) {
        if (update_model_gradients) {
          p_grad_updater_bias->update(_cpu_bridges[0]->get_bias_grad_cube()->get_p_data());
        }
        // Save this pointer for later
        pointer_to_host_copy_of_latest_bias_grad = _cpu_bridges[0]->get_bias_grad_cube()->get_p_data();
      } else {
        // Here handle the special-case of a single GPU. The GPU can
        // sum its own gradient to its model and therefore we can eliminate a copy
        // of the gradients back to the host, the CPU saxpy updates, and the copy
        // of the new model back to the device.
#ifdef _INCLUDE_GPUDRIVER
    #ifdef _DO_ASSERT
        assert(skip_model_copy_gpu);
    #endif
        if (update_model_gradients) {
          gpu_grad_updater_bias->update(_gpu_bridges[0]->get_bias_grad_cube()->get_p_data());
        }
        // Save this pointer for later (NULL since on device)
        pointer_to_host_copy_of_latest_bias_grad = NULL;
#endif
      }
    }
  }

  PROFILE_ONLY(seconds = t.elapsed(); std::cout << "    PB:  Update bias:             " << seconds << "\n"; t.restart(); )

  // Now that the backward step is complete, copy data gradients back to the host
  // This is analogous to the copy earlier in this function from host -> device
  if (needs_to_calc_backward_grad && !share_pointer_with_prev_bridge)
  {
      if (num_partitions_GPU == 1) { // Special case: Don't launch threads
        // Get the amount to copy from GPU
        size_t n_batch_this_partition = GPU_batch_sizes[0];
#ifdef _DO_ASSERT
        assert(n_batch_this_partition > 0);
#endif
        size_t b = num_partitions_CPU * n_batch_per_partition_cpu;
        
        // Copy from the GPU's output cube back to the location b of the PBridge's data/gradient layers
        scheduler_gpudrivers[0]->memcpy(
            // dst = the cpu data in the output layer starting at the right batch (batch "b")
            p_input_layer->p_gradient_cube->get_device_pointer_RCDslice(scheduler_local_cpudriver, b, n_batch_this_partition),
            // src = the cube in the _data_cubes_higher vector corresponding to this GPU
            // (this is where the output data for the GPU was stored during the bridge execution)
            _grad_cubes_lower[0 + num_partitions_CPU]->get_device_pointer(scheduler_gpudrivers[0])
        );
        // Update the pointer to the right batch for the next sub-bridge
        // b += n_batch_this_partition;
      } else if (num_partitions_GPU > 1) {
          vector<thread> threads;
          for (size_t gpu_i = 0; gpu_i < num_partitions_GPU; ++gpu_i)
          {
            threads.push_back(thread([this, gpu_i]() {
                // Get the amount to copy from GPU
                size_t n_batch_this_partition = GPU_batch_sizes[gpu_i];
        #ifdef _DO_ASSERT
                assert(n_batch_this_partition > 0);
        #endif
                size_t b = num_partitions_CPU * n_batch_per_partition_cpu;
                for (size_t bi=0; bi<gpu_i; ++bi) {
                    b += GPU_batch_sizes[bi];
                }
                
                // Copy from the GPU's output cube back to the location b of the PBridge's data/gradient layers
                scheduler_gpudrivers[gpu_i]->memcpy(
                    // dst = the cpu data in the output layer starting at the right batch (batch "b")
                    p_input_layer->p_gradient_cube->get_device_pointer_RCDslice(scheduler_local_cpudriver, b, n_batch_this_partition),
                    // src = the cube in the _data_cubes_higher vector corresponding to this GPU
                    // (this is where the output data for the GPU was stored during the bridge execution)
                    _grad_cubes_lower[gpu_i + num_partitions_CPU]->get_device_pointer(scheduler_gpudrivers[gpu_i])
                );
                // Update the pointer to the right batch for the next sub-bridge
                // b += n_batch_this_partition;
            }));
          }
          for (size_t ti = 0; ti < threads.size(); ti++) {
            threads[ti].join();
          }
      }
  }
  // SHADJIS TODO: It makes sense to sync here so we do not return too early
  // before the computations are done. Syncing is automatically done by copying
  // back to the host but since sometimes we skip this, I'll add a sync here for
  // each device. Might not be necessary though.
  else {
      for (size_t gpu_i = 0; gpu_i < num_partitions_GPU; ++gpu_i) {
        scheduler_gpudrivers[gpu_i]->device_sync();
      }
  }
  PROFILE_ONLY(seconds = t.elapsed(); std::cout << "    PB:  Copy Data -> Host:       " << seconds << "\n"; t.restart(); )

  if (skip_model_copy_gpu) {
    scheduler_gpudrivers[0]->destroy_thread();
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

  if (p_model_cube)    delete p_model_cube;
  if (p_model_grad)    delete p_model_grad;
  for (size_t pi=0; pi<p_model_subgrads.size(); ++pi) {
    delete p_model_subgrads[pi];
  }
  if (p_grad_updater)  delete p_grad_updater;
  if (p_bias_cube)     delete p_bias_cube;
  if (p_bias_grad)     delete p_bias_grad;
  if (p_bias_subgrad)  delete p_bias_subgrad;
#ifdef _INCLUDE_GPUDRIVER
  if (gpu_grad_updater)  delete gpu_grad_updater;
  if (gpu_grad_updater_bias) delete gpu_grad_updater_bias;
#endif  
  // SHADJIS TODO: Delete these
  // delete scheduler_local_cpudriver;
  // if (scheduler_gpudriver) { delete scheduler_gpudriver; }
}

#endif
