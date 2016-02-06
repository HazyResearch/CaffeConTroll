//
//  ParallelizedBridge.h
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_ParallelizedBridge_h
#define moka_ParallelizedBridge_h

#include "AbstractBridge.h"
#include "PhysicalStratum.h"
#include <thread>
#include <vector>

using std::vector;

// For now, we only support Layout_CRDB
// A ParallelizedBridge is an AbstractBridge on the CPU
// However, it also has other internal drivers for creating other bridges on these devices
// So a ParallelizedBridge is an AbstractBridge, and a ConvolutionBridge is an AbstractBridge,
// but a ParallelizedBridge may create a new ConvolutionBridge.

template<// The first template for the ParallelizedBridge is just the type
         typename DataType, 
         
         // The second template for the ParallelizedBridge is the bridge type, e.g. ConvolutionBridge
         // However, a ConvolutionBridge is not a class, it is a template class
         // (E.g. it is not a vector<int>, it is a vector)
         // So we need to declare as a template:
         template
         // Next, we need to templatize the same way as the bridge
         // A bridge takes the following template arguments:
          <typename InputLayerDataType, LayoutType InputLayerLayout,
           typename OutputLayerDataType, LayoutType OutputLayerLayout,
           typename DriverClass> 
         // Note above: If we want to pass in a value, like 5, then we use template <int>
         // If we want to pass in a type, like int, then we use template <class T>
         // (class is identical to typename)
         // So InputLayerDataType and OutputLayerDataType are types, like float
         // The DriverClass is also a type
         // But LayoutType is a value, like Layout_CRDB (note, this can change)
         // Also the name of the type can be omitted
         //
         // Then, finally include the BridgeType
         class BridgeType>
class ParallelizedBridge : public AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver> {
  protected:
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::curr_B;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::input_d_cube;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::input_g_cube;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::output_d_cube;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::output_g_cube;

  public:
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::report_forward_constructor;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::report_forward_last_transfer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::report_forward_history;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::run_with_n_threads;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::report_backward_updateweight_constructor;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::report_backward_updateweight_last_transfer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::report_backward_updateweight_history;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::needs_to_calc_backward_grad;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::layer_param;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::solver_param;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::p_driver;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::p_input_layer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver>::p_output_layer;

    typedef LogicalCube<DataType, Layout_CRDB> LogicalCubeType;

    typedef Layer<DataType, Layout_CRDB> LayerType;

    // These are public for now, just so that we can write tests
    LogicalCubeType * p_model_cube; /** A ParallelizedConvolutionBridge should have a _single_
                                        copy of the model. Copy this model to different worker (or
                                        add optimization to share without copying) is the job
                                        of ParallelizedConvolutionBridge not its caller. **/
    LogicalCubeType * p_model_grad;
    std::vector<LogicalCubeType *> p_model_subgrads;  // One per partition
    LogicalCubeType * p_bias_grad;
    LogicalCubeType * p_bias_subgrad;
    LogicalCubeType * p_bias_cube;
    
    
    // -------------------------------------------------------------------------
    // Scheduler class members (class members related to partition scheduling)
    // SHADJIS TODO: Make these private
    // -------------------------------------------------------------------------
    
    // Note: Previously some of these weren't global class members and
    // were instead always re-calculated. I think that was done to handle
    // different grouping but that should be abstracted to the pbridge so
    // it only needs to worry about its current group.
    
    // Inputs (never change)
    const size_t n_partition; // requested # CPU partitions, not counting extra partition
    const size_t n_batch; // batch size of regular batches (excludes last one)
    const size_t n_cpu_thread_per_partition; // Number of CPU threads for OpenBLAS
    
    // The rest are calculated based on inputs above:
    
    // Scheduling for multiple devices
    
    // For CPU-only, CcT takes a batch size (curr_B), a number of partitions (n_partition),
    // and equally divides to obtain n_batch_per_partition_cpu. Then, 1 thread is launched per
    // partition with a batch size of n_batch_per_partition_cpu (plus maybe 1 extra for 
    // leftover images). For GPU however, we want to have a single partition (i.e.
    // 1 CPU thread), but with a larger batch size. 
    //
    // Example
    //
    // To make this clear let's use an example
    // Say n_partition = 16 above (e.g. we have 16 cores or threads) and n_batch = 260. 
    // (n_cpu_thread_per_partition isn't important).
    // Then, extra_partition is true (since one partition must have 260-16*16=4)
    //
    // Example 1: Everything on CPU
    // We need 17 CPU partitions, if everything is on the CPU.
    // Then,
    //   extra_partition = true
    //   num_partitions_CPU = 17
    //   num_partitions_GPU = 0
    //   num_partitions = 17
    //   n_batch_per_partition_cpu = 16 (16*16 = 256, ignoring last partition)
    //
    // Example 2: 50% on GPU
    // Now, of the 260 images, 50% will be on the GPU, 1 big partition. However,
    // rather than split 130 / 130, we want the remaining CPU images to equally,
    // so instead we will split 128 vs. 132: the CPU takes 128 ( = 16 * 8), and
    // the GPU takes the remaining 132:
    // So:
    //   extra_partition = false (leftovers go on the GPU)
    //   num_partitions_CPU = 16 (i.e. still use all 16 cores as always)
    //   num_partitions_GPU = 1
    //   num_partitions = 17 (=16+1)
    //   n_batch_per_partition_cpu = 8 (16*8 = 128)
    // and GPU_batch_sizes would be a vector of size 1, containing the number 260 - 128 = 132.
    //
    // In general, you can check the number of devices in cuda using 
    // cudaGetDeviceCount, or in the terminal, nvidia-smi.
    // SHADJIS TODO: #ifdef _INCLUDE_GPUDRIVER, include cuda runtime .h and check
    // cudaGetDeviceCount when initializing GPU_batch_sizes.
    //
    // Any leftover in the last partition will be handled by whichever device
    // is executing the last normal partition.
    
    bool extra_partition; // Do we need to handle an extra partitions
    size_t num_partitions_CPU;
    size_t num_partitions_GPU; // Number of partitions across ALL GPUs
    size_t num_partitions; // #CPU partitions + #GPU partitions (including extra)
    size_t n_batch_per_partition_cpu; // Batch size for cpu partitions (excluding extra)
    std::vector <size_t> GPU_batch_sizes; // Batch sizes for gpu partitions (may differ per GPU)
    // SHADJIS TODO: Remove this. I am handling a corner-case where there are n GPUs
    // in the system, but not all are in use for some reason, AND the used GPUs are
    // not consecutive from the beginning. E.g. you have GPU 0-3 in your system but
    // only want to sue GPU2 for some reason.
    std::vector <int> used_gpu_to_device_id_map;
    
    // SHADJIS TODO: Can return a (const?) reference instead
    std::vector <size_t> get_GPU_batch_sizes() { return GPU_batch_sizes; }
    std::vector <int> get_used_gpu_to_device_id_map() { return used_gpu_to_device_id_map; }
    std::vector<LogicalCubeType *> get_data_cubes_higher() { return _data_cubes_higher; }
    std::vector<LogicalCubeType *> get_grad_cubes_higher() { return _grad_cubes_higher; }
    size_t get_num_partitions_CPU() { return num_partitions_CPU; }
    
    // A local CPU driver used by the scheduler
    // This is the same driver which templatizes the ParallelizedBridge,
    // and is used e.g. for collecting gradients
    CPUDriver * scheduler_local_cpudriver;
#ifdef _INCLUDE_GPUDRIVER
    // The GPU Driver, can add more drivers here to put into a vector
    // SHADJIS TODO: This should be a vector of GPUDrivers, and each will be given
    // a stream # and device # as an argument (for now, keep it to a single driver)
    std::vector<GPUDriver *> scheduler_gpudrivers;
#else
    // SHADJIS TODO: If _INCLUDE_GPUDRIVER is undefined just make this a cpu driver
    // and assert it's never used. That way I don't have to ifdef it out everywhere.
    std::vector<CPUDriver *> scheduler_gpudrivers;
#endif

    // Also, keep track of whether this PBridge shares pointers to data and
    // gradients with the bridges that came before and after it. Originally
    // this was always true since every PBridge began by copying its data from
    // the CPU and finished by copying its data back to the CPU. So everything 
    // was always on the CPU in between. Since this is inneficient, instead we
    // need to keep track of whether this bridge shares pointers with the bridges
    // which come before and after it, or equivalently whether it is necessary
    // to copy from / back to the CPU (for sub-bridges on the CPU this makes no
    // difference since it is already on the CPU).
    bool share_pointer_with_prev_bridge;
    bool share_pointer_with_next_bridge;
    // Some bridges, e.g. ReLU and dropout, have the same top and bottom layer,
    // i.e. they do not need to allocate any cubes
    bool share_input_output_layer;
    // For the GPU, we might not need to copy the model back to the host
    bool skip_model_copy_gpu;
    
    int model_parallelism_group_size;
    
    // -------------------------------------------------------------------------
    // End of Scheduler class members
    // -------------------------------------------------------------------------

    float model_base_learning_rate;
    float bias_base_learning_rate;
    float model_base_regularization;
    float bias_base_regularization;
    
    // If doing asynchronous model updates, we do not always want the model
    // update to take place immediately after the backward pass. This can be
    // the case if we want a shared model on some server with other servers
    // passing back gradients (parameter server).
    bool update_model_gradients;

    // Also keep pointers to where the gradients are after the backwards step
    // This is in case we choose not to update the gradients in the backwards
    // step but rather store them to be read later.
    // 
    // There are many ways to accomplish that. For example, each backwards pass,
    // we can either return 2 pointers: to the accumulated model gradient, and
    // to the accumulated bias gradient. Then the gradients can be copied to some
    // other location, etc.
    // 
    // Another strategy is to just store those pointers to the most recently
    // computed gradients in this pbridge class, and then expose a function
    // which returns a pointer to those gradients when they are needed. That
    // is the one I will do now.
    //
    // There are many optimizations around accumulating gradients inside the
    // pbridge. Overall we have gradients for model and for bias, as well as
    // 4 cases of gradient accumulation:
    //
    //  - #bridges > 1, at least one on CPU
    //  - #bridges > 1, all on GPU
    //  - #bridges = 1, on CPU
    //  - #bridges = 1, on GPU
    //
    // So overall we have 8 cases (4 for model gradient, 4 for bias gradient)
    // Currently, these are the locations of the gradient following a call
    // to backward():
    //
    // Model:
    //  - #bridges > 1, at least one on CPU
    //      - _cpu_bridges[0]->get_model_grad_cube()->get_p_data()
    //      - This memory is owned by the sub-bridge (host)
    //  - #bridges > 1, all on GPU
    //      - p_model_subgrads[0]
    //      - This memory is owned by the parallelized bridge (host)
    //  - #bridges = 1, on CPU
    //      - _cpu_bridges[0]->get_model_grad_cube()->get_p_data()
    //      - This memory is owned by the sub-bridge (host)
    //  - #bridges = 1, on GPU
    //      - _gpu_bridges[0]->get_model_grad_cube()->get_p_data()
    //      - This memory is owned by the sub-bridge (device, never copied back to host)
    // 
    // Bias:
    //  - #bridges > 1, at least one on CPU
    //      - p_bias_grad->get_p_data()
    //      - This memory is owned by the parallelized bridge (host)
    //  - #bridges > 1, all on GPU
    //      - p_bias_grad->get_p_data()
    //      - This memory is owned by the parallelized bridge (host)
    //  - #bridges = 1, on CPU
    //      - _cpu_bridges[0]->get_bias_grad_cube()->get_p_data()
    //      - This memory is owned by the sub-bridge (host)
    //  - #bridges = 1, on GPU
    //      - _gpu_bridges[0]->get_bias_grad_cube()->get_p_data()
    //      - This memory is owned by the sub-bridge (device, never copied back to host)
    //
    // Side note: The reason for asymmetry in case 1 and 2 for model vs bias is that for the 
    // model case we employ the following optimization:
    // For bias, which is small, we (1) keep a host copy of the bias gradient, (2) set it to 0,
    // then iterate over each partition's gradients and add them to the bias gradient from (1).
    // For model, we save one of these accumulations by getting a pointer to the first partition's
    // gradient (which is always on host in case 1 or is copied to host in case 2) and then iterate
    // over each other partition's gradients (i.e. starting from the 2nd one) and add to the first.
    // For large models (fully-connected) the speedup was not insignificant.
    // Another difference for model vs bias is that for multiple GPUs we copy back to host in parallel
    // whereas for bias we don't. This is why PBridge has only one buffer cube for bias gradients,
    // i.e. p_bias_grad, whereas for model we have p_model_subgrads.
    //
    // So given that all these cases store gradients in different places, let's just introduce 2 
    // pointers and set them to wherever the gradients are stored. If the gradients are on the
    // device, then we know we are in case 4 and so we can return NULL.
    DataType * pointer_to_host_copy_of_latest_model_grad;
    DataType * pointer_to_host_copy_of_latest_bias_grad;
    

    // For now, run the gradient updates on the CPU
    // See comment in ParallelizedBridge_impl.hxx
    GradientUpdater<DataType, CPUDriver> * p_grad_updater;
    GradientUpdater<DataType, CPUDriver> * p_grad_updater_bias;
#ifdef _INCLUDE_GPUDRIVER
    // Update: Eventually we want to distribute updates for all GPUs.
    // For now this is only handled for the case of a single GPU.
    GradientUpdater<DataType, GPUDriver> * gpu_grad_updater;
    GradientUpdater<DataType, GPUDriver> * gpu_grad_updater_bias;
#endif


    ParallelizedBridge(Layer<DataType, Layout_CRDB> * const _input_layer,
        Layer<DataType, Layout_CRDB> * const _output_layer,
        const cnn::LayerParameter * const _layer_param,
        const cnn::SolverParameter * const _solver_param,
        CPUDriver * const _p_driver, size_t _n_partition,
        size_t _n_cpu_thread_per_partition,
        const size_t PREVIOUS_BRIDGE_num_partitions_CPU = 0,
        const std::vector<size_t>& PREVIOUS_BRIDGE_GPU_batch_sizes = std::vector<size_t>(),
        const std::vector<int>   & PREVIOUS_BRIDGE_used_gpu_to_device_id_map = std::vector<int>(),
        const std::vector<LogicalCubeType *> & PREVIOUS_BRIDGE_data_cubes_lower  = std::vector<LogicalCubeType *>(),
        const std::vector<LogicalCubeType *> & PREVIOUS_BRIDGE_grad_cubes_lower  = std::vector<LogicalCubeType *>(),
        bool _share_input_output_layer = false);

    ~ParallelizedBridge();

    void forward();

    void backward();

    LogicalCube<DataType, Layout_CRDB> * const get_model_cube() {
        return p_model_cube;
    }

    LogicalCube<DataType, Layout_CRDB> * const get_bias_cube() {
        return p_bias_cube;
    }

    // This shouldn't be used because the updater will be NULL if there is
    // only 1 bridge and it is a GPU bridge
    GradientUpdater<DataType, CPUDriver> * const get_model_updater() {
        return p_grad_updater;
    }

    // This shouldn't be used because the updater will be NULL if there is
    // only 1 bridge and it is a GPU bridge
    GradientUpdater<DataType, CPUDriver> * const get_bias_updater() {
        return p_grad_updater_bias;
    }
    
    void set_share_pointer_with_prev_bridge(bool _share) {
        share_pointer_with_prev_bridge = _share;
    }
    bool get_share_pointer_with_prev_bridge() {
        return share_pointer_with_prev_bridge;
    }
    void set_share_pointer_with_next_bridge(bool _share) {
        share_pointer_with_next_bridge = _share;
    }
    bool get_share_pointer_with_next_bridge() {
        return share_pointer_with_next_bridge;
    }

    void set_update_model_gradients(bool _update_model_gradients) {
        update_model_gradients = _update_model_gradients;
    }
    
    int get_model_parallelism_group_size() { return model_parallelism_group_size; }
    void set_model_parallelism_group_size(int _model_parallelism_group_size) {
        model_parallelism_group_size = _model_parallelism_group_size;
    }
    
    // SHADJIS TODO: May be possible to have a single vector for all bridges
    // These can also be protected like they used to be
    vector<BridgeType <DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver> *> _cpu_bridges;
#ifdef _INCLUDE_GPUDRIVER
    vector<BridgeType <DataType, Layout_CRDB, DataType, Layout_CRDB, GPUDriver> *> _gpu_bridges;
#else
    // SHADJIS TODO: If _INCLUDE_GPUDRIVER is undefined just make this a cpu driver
    // and assert it's never used. That way I don't have to ifdef it out everywhere.
    vector<BridgeType <DataType, Layout_CRDB, DataType, Layout_CRDB, CPUDriver> *> _gpu_bridges;
#endif
    
    // Recently I added an optimization to not have to copy the model cubes back to the host all the
    // time (since it is slow for fully connected), but to keep on device. However, sometimes we
    // want to write the model to file or read it from file, so it must be on the host.
    // We can force that copy with these
    // SHADJIS TODO: Should check here if the src pointer is NULL. If so, don't copy.
    void force_host_to_device_model_copy() {
        if (skip_model_copy_gpu) {
            scheduler_gpudrivers[0]->memcpy(_gpu_bridges[0]->get_model_cube()->get_device_pointer(scheduler_gpudrivers[0]), p_model_cube->get_device_pointer(scheduler_local_cpudriver));
        }
    }
    void force_host_to_device_bias_copy()  {
        if (skip_model_copy_gpu) {
            scheduler_gpudrivers[0]->memcpy(_gpu_bridges[0]->get_bias_cube() ->get_device_pointer(scheduler_gpudrivers[0]), p_bias_cube ->get_device_pointer(scheduler_local_cpudriver));
        }
    }
    void force_device_to_host_model_copy() {
        if (skip_model_copy_gpu) {
            scheduler_gpudrivers[0]->memcpy(p_model_cube->get_device_pointer(scheduler_local_cpudriver), _gpu_bridges[0]->get_model_cube()->get_device_pointer(scheduler_gpudrivers[0]));
        }
    }
    void force_device_to_host_bias_copy()  {
        if (skip_model_copy_gpu) {
            scheduler_gpudrivers[0]->memcpy(p_bias_cube->get_device_pointer(scheduler_local_cpudriver), _gpu_bridges[0]->get_bias_cube()->get_device_pointer(scheduler_gpudrivers[0]));
        }
    }
    

    // These return a host pointer to the last model gradients calculated by this bridge
    DataType * get_model_gradient_host() {
        
        if (pointer_to_host_copy_of_latest_model_grad) {
            return pointer_to_host_copy_of_latest_model_grad;
        }
        // There are two reasons this is NULL:
        //  1. This is on GPU
        //  2. This has not been calculated yet
        else if (skip_model_copy_gpu) {
            scheduler_gpudrivers[0]->memcpy(p_model_grad->get_device_pointer(scheduler_local_cpudriver), _gpu_bridges[0]->get_model_grad_cube()->get_device_pointer(scheduler_gpudrivers[0]));
            return p_model_grad->get_p_data();
        }
        // This has not been calculated yet, must return NULL
        return NULL;
    }
    DataType * get_bias_gradient_host()  {

        if (pointer_to_host_copy_of_latest_bias_grad) {
            return pointer_to_host_copy_of_latest_bias_grad;
        }
        // There are two reasons this is NULL:
        //  1. This is on GPU
        //  2. This has not been calculated yet
        else if (skip_model_copy_gpu) {
            scheduler_gpudrivers[0]->memcpy(p_bias_grad->get_device_pointer(scheduler_local_cpudriver), _gpu_bridges[0]->get_bias_grad_cube()->get_device_pointer(scheduler_gpudrivers[0]));
            return p_bias_grad->get_p_data();
        }
        // This has not been calculated yet, must return NULL
        return NULL;
    }


    // Update the model or bias given a gradient
    void update_model_with_gradient_CPU(float * grad) {
    
        // If this is a GPU gradient, we need to call:
        //    gpu_grad_updater->update(grad);
        // but then we need to pass in a device pointer to this function
        // For now just assume gradient update is on CPU
        assert (!skip_model_copy_gpu);
        p_grad_updater->update(grad);
    }
    void update_bias_with_gradient_CPU(float * grad) {
        
        //if (skip_model_copy_gpu)
        //    gpu_grad_updater_bias->update(grad);
        assert (!skip_model_copy_gpu);
        p_grad_updater_bias->update(grad);
    }
    
    

    // Update the pointer of the input layer's data
    // This function is used to run a different dataset on the network
    // (e.g. for validation set)
    void update_p_input_layer_data_CPU_ONLY(float * new_data)  {
      // Step 1: Update the input layer for the pbridge
      p_input_layer->p_data_cube->set_p_data(new_data);
      // Step 2: Update the input layer for each sub-bridge
      // Note: we only need to do this for CPU bridges because GPU
      // bridges use pointers to device memory (not the corpus data,
      // which is on the host)
      // SHADJIS TODO: This will only work on GPU for the first layer
      // If we want to update the input data for other layers we need
      // to do a copy to device memory
      for (size_t i = 0, b = 0; i < _cpu_bridges.size(); ++i, b += n_batch_per_partition_cpu) {
        _cpu_bridges[i]->update_p_input_layer_data_CPU_ONLY(p_input_layer->p_data_cube->physical_get_RCDslice(b));
      }
    }
    
    // Update the pointer of the output layer's gradient
    // This function is used to run a different dataset on the network
    // (e.g. for validation set)
    void update_p_output_layer_gradient_CPU_ONLY(float * new_data)  {
      // Step 1: Update the output layer for the pbridge
      p_output_layer->p_gradient_cube->set_p_data(new_data);
      // Step 2: Update the output layer for each sub-bridge
      // See TODO in AbstractBridge.h for this function to see what to change
      // if we want to update the gradients for output gradients on the GPU
      for (size_t i = 0, b = 0; i < _cpu_bridges.size(); ++i, b += n_batch_per_partition_cpu) {
        _cpu_bridges[i]->update_p_output_layer_gradient_CPU_ONLY(p_output_layer->p_gradient_cube->physical_get_RCDslice(b));
      }
    }
    
  protected:

    // Overload this for PBridge  
    // If curr_B changes, also need to update partitions
    virtual void set_curr_batch_size(const size_t _curr_B) {
      if (curr_B == _curr_B) {
        return;
      }
      curr_B = _curr_B;
      update_scheduler_partitions_for_curr_B();
    }
    
    // When the batch size changes (only for the last batch which may be smaller),
    // update how we distribute
    void update_scheduler_partitions_for_curr_B() {

      assert(curr_B <= n_batch);
      
      // Sum up the amount each GPU is going to take from this batch
      std::vector <float> GPU_batch_proportions;
      // SHADJIS TODO: I am hard-coding 4 now. Eventually we want to 
      // specify the number somewhere, e.g. have a separate section in
      // the prototxt. We can also abstract this all from the user and
      // just call cudaGetDeviceCount (the scheduler can do this eventually)
      GPU_batch_proportions.push_back(layer_param->gpu_0_batch_proportion());
      GPU_batch_proportions.push_back(layer_param->gpu_1_batch_proportion());
      GPU_batch_proportions.push_back(layer_param->gpu_2_batch_proportion());
      GPU_batch_proportions.push_back(layer_param->gpu_3_batch_proportion());
      
      // Determine GPU batch sizes
      num_partitions_GPU = 0;
      std::vector <size_t> GPU_batch_sizes_tmp;
      std::vector <int> used_gpu_to_device_id_map_tmp;
      size_t total_GPU_batch_size = 0;
      for (size_t i=0; i<GPU_batch_proportions.size(); ++i) {
        size_t num_on_this_gpu = GPU_batch_proportions[i] * curr_B;
        if (num_on_this_gpu > 0) {
          GPU_batch_sizes_tmp.push_back(num_on_this_gpu);
          used_gpu_to_device_id_map_tmp.push_back(i);
          total_GPU_batch_size += num_on_this_gpu;
          ++num_partitions_GPU;
        }
      }
      assert(total_GPU_batch_size <= curr_B);
      GPU_batch_sizes = GPU_batch_sizes_tmp;
      used_gpu_to_device_id_map = used_gpu_to_device_id_map_tmp;
    #ifndef _INCLUDE_GPUDRIVER
      if (num_partitions_GPU > 0) {
        std::cout << "\nError, GPU not enabled. To enable running on GPU add NVCC and CUDA_INCLUDE to .config\n\n";
        assert(false);
      }
    #endif
      
      // Remainder goes on the CPU
      size_t remainder = curr_B - total_GPU_batch_size;
      // There are 2 cases: Either what is left is less than the number of cpu
      // partitons, or not. 
      
      // In this case, just assign all remaining images to 1 partition each
      if (remainder < n_partition) {
        num_partitions_CPU = remainder;
        if (num_partitions_CPU > 0) {
          n_batch_per_partition_cpu = remainder / num_partitions_CPU;
        } else {
          n_batch_per_partition_cpu = 0;
        }
        extra_partition = false;
      }
      // In this case, we have more images than partitions.
      // It could be that they divide perfectly, but if not we need to handle an extra partition.
      // We do this differently depending on whether a GPU is being used:
      // - If using at least 1 GPU, give it the extra partition data
      //   (SHADJIS TODO: Can fix this if it's a problem, currently decided arbitrarily)
      // - If not, assign 1 extra CPU partition
      else {
        // Check if there is an extra partition
        if (total_GPU_batch_size > 0) {
          assert(num_partitions_GPU > 0);
          // Assign the extra partition to the GPU
          extra_partition = false;
          num_partitions_CPU = n_partition;
          n_batch_per_partition_cpu = remainder / num_partitions_CPU;
          const size_t extra_partition_size = remainder % num_partitions_CPU;
          // Assign extra partition equally to all GPUs
          const size_t extra_partition_size_per_gpu = extra_partition_size / num_partitions_GPU;
          const size_t extra_partition_size_gpu_0   = extra_partition_size_per_gpu + extra_partition_size % num_partitions_GPU;
          assert(GPU_batch_sizes[0] >= 0);
          GPU_batch_sizes[0] += extra_partition_size_gpu_0;
          total_GPU_batch_size += extra_partition_size_gpu_0;
          for (size_t gpu_i=1; gpu_i < GPU_batch_sizes.size(); ++gpu_i) {
            assert(GPU_batch_sizes[gpu_i] >= 0);
            GPU_batch_sizes[gpu_i] += extra_partition_size_per_gpu;
            total_GPU_batch_size += extra_partition_size_per_gpu;
          }
          assert(total_GPU_batch_size <= curr_B);
        }
        // The batch is entirely on the CPU, so any extra images 
        // require an extra partition
        else {
          extra_partition = curr_B % n_partition > 0;
          n_batch_per_partition_cpu = remainder / n_partition; // Excluding last batch
          num_partitions_CPU = extra_partition ? n_partition + 1: n_partition;
        }
      }
      num_partitions = num_partitions_CPU + num_partitions_GPU;
      assert(num_partitions > 0);
      assert(used_gpu_to_device_id_map.size() == num_partitions_GPU);
      assert(n_batch_per_partition_cpu + total_GPU_batch_size > 0);

      // std::cout << "  #Partitions Total    = " << num_partitions << "\n";
      // std::cout << "  #Partitions on CPU   = " << num_partitions_CPU << "\n";
      // std::cout << "  #Partitions on GPU   = " << num_partitions_GPU << "\n";
      // std::cout << "  Total batch size GPU = " << total_GPU_batch_size << "\n";
      // for (int it = 0; it < num_partitions_GPU; ++it) {
      //   std::cout << "      " << GPU_batch_sizes[it] << "\n";
      // }
      // std::cout << "  Partition size CPU   = " << n_batch_per_partition_cpu << "\n";
      // std::cout << "  Extra CPU partition  = " << extra_partition << "\n";
    }
  
    vector<LogicalCubeType *> _data_cubes_lower;
    vector<LogicalCubeType *> _grad_cubes_lower;

    vector<LogicalCubeType *> _data_cubes_higher;
    vector<LogicalCubeType *> _grad_cubes_higher;

    vector<LayerType *> _partitioned_layers_lower;
    vector<LayerType *> _partitioned_layers_higher;


    PhysicalStratum stratum;
    
    // Helper function for pbridge, checks if vectors match
    bool do_bridges_have_same_device_assignments(
        const std::vector<size_t>& GPU_batch_sizes = std::vector<size_t>(),
        const std::vector<int>   & used_gpu_to_device_id_map = std::vector<int>(),
        const std::vector<size_t>& PREVIOUS_BRIDGE_GPU_batch_sizes = std::vector<size_t>(),
        const std::vector<int>   & PREVIOUS_BRIDGE_used_gpu_to_device_id_map = std::vector<int>())
    {
        // First check if the sizes mismatch
        if (GPU_batch_sizes.size() != PREVIOUS_BRIDGE_GPU_batch_sizes.size()) {
            return false;
        }
        if (used_gpu_to_device_id_map.size() != PREVIOUS_BRIDGE_used_gpu_to_device_id_map.size()) {
            return false;
        }
        // We know the same # GPUs are used for both bridges
        // Next, check if the GPUs are the same
        for (size_t i=0; i<used_gpu_to_device_id_map.size(); ++i) {
            if (used_gpu_to_device_id_map[i] != PREVIOUS_BRIDGE_used_gpu_to_device_id_map[i]) {
                return false;
            }
        }
        // The GPUs match. Finally, check that the batch sizes match.
        for (size_t i=0; i<GPU_batch_sizes.size(); ++i) {
            if (GPU_batch_sizes[i] != PREVIOUS_BRIDGE_GPU_batch_sizes[i]) {
                return false;
            }
        }
        // The previous pbridge and the current one allocate batches identically to CPU/GPU, so
        // we can share the device pointers.
        // SHADJIS TODO: In the future less strict conditions may apply, e.g. if we go from 2 GPUs
        // to 1 GPU, we can still share some of the pointers (e.g. half of the data is still on GPU 1)
        return true;
    }

};

#include "ParallelizedBridge_impl.hxx"

#endif
