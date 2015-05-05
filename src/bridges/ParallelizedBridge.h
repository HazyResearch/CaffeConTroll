//
//  ParallelizedBridge.h
//  moka
//
//  Created by Firas Abuzaid on 2/8/15.
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
    LogicalCubeType * p_model_subgrad;
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
    
    // Calculated based on inputs
    
    // Example
    //
    // To make this clear let's use an example
    // Say n_partition = 16 above and n_batch = 260. 
    // (n_cpu_thread_per_partition isn't important).
    // Then, extra_partition is true (since one partition must have 260-16*16=4)
    //
    // Example 1: Everything on CPU
    // We need 17 CPU partitions, if everything is on the CPU.
    // Then,
    //   num_partitions_CPU = 17
    //   num_partitions_GPU = 0
    //   num_partitions = 17
    //   n_batch_per_partition_cpu = 16 (16*16 = 256, ignoring last partition)
    //   n_batch_per_partition_gpu = 0 (irrelevant)
    //
    // Example 2: 50% on GPU
    // Now, of the 17 CPU partitions, only 8 will be on the CPU, the
    // rest will be on the GPU. But the GPU just gets 1 big partition,
    // So:
    //   num_partitions_CPU = 8
    //   num_partitions_GPU = 1
    //   num_partitions = 9
    //   n_batch_per_partition_cpu = 16 (16*16 = 256, ignoring last partition)
    //   n_batch_per_partition_gpu = 0 (irrelevant)
    //
    // Any leftover in the last partition will be handled by whichever device
    // is executing the last normal partition.
    
    bool extra_partition; // Do we need to handle an extra partitions
    size_t num_partitions_CPU;
    size_t num_partitions_GPU; // Number of partitions across ALL GPUs
    size_t num_partitions; // #CPU partitions + #GPU partitions (including extra)
    size_t n_batch_per_partition_cpu; // Batch size for cpu partitions (excluding extra)
    size_t n_batch_per_partition_gpu; // Batch size for gpu partitions
    
    // A local CPU driver used by the scheduler
    // This is the same driver which templatizes the ParallelizedBridge,
    // and is used e.g. for collecting gradients
    CPUDriver * scheduler_local_cpudriver;
#ifdef _INCLUDE_GPUDRIVER
    // The GPU Driver, can add more drivers here to put into a vector
    // SHADJIS TODO: This should be a vector of GPUDrivers, and each will be given
    // a stream # and device # as an argument (for now, keep it to a single driver)
    GPUDriver * scheduler_gpudriver;
#else
    // SHADJIS TODO: If _INCLUDE_GPUDRIVER is undefined just make this a cpu driver
    // and assert it's never used. That way I don't have to ifdef it out everywhere.
    CPUDriver * scheduler_gpudriver;
#endif
    // -------------------------------------------------------------------------
    // End of Scheduler class members
    // -------------------------------------------------------------------------

    float model_base_learning_rate;
    float bias_base_learning_rate;
    float model_base_regularization;
    float bias_base_regularization;

    // For now, run the gradient updates on the CPU
    // See comment in ParallelizedBridge_impl.hxx
    GradientUpdater<DataType, CPUDriver> * p_grad_updater;
    GradientUpdater<DataType, CPUDriver> * p_grad_updater_bias;

    ParallelizedBridge(Layer<DataType, Layout_CRDB> * const _input_layer,
        Layer<DataType, Layout_CRDB> * const _output_layer,
        const cnn::LayerParameter * const _layer_param,
        const cnn::SolverParameter * const _solver_param,
        CPUDriver * const _p_driver, size_t _n_partition,
        size_t _n_cpu_thread_per_partition);

    ~ParallelizedBridge();

    void forward();

    void backward();

    LogicalCube<DataType, Layout_CRDB> * const get_model_cube() {
        return p_model_cube;
    }

    LogicalCube<DataType, Layout_CRDB> * const get_bias_cube() {
        return p_bias_cube;
    }

    GradientUpdater<DataType, CPUDriver> * const get_model_updater() {
        return p_grad_updater;
    }

    GradientUpdater<DataType, CPUDriver> * const get_bias_updater() {
        return p_grad_updater_bias;
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
    
    void update_scheduler_partitions_for_curr_B() {

      assert(curr_B <= n_batch);

      // Check if n_partition is greater than the actual batch size
      const size_t n_partition_this_batch = (curr_B < n_partition) ? curr_B : n_partition;
      n_batch_per_partition_cpu = curr_B / n_partition_this_batch; // Excluding last batch
      
      // Adjust the number of partitions if we need an extra one
      extra_partition = curr_B % n_partition_this_batch > 0;
      const size_t total_num_partitions_if_all_cpu =  extra_partition ? n_partition_this_batch + 1: n_partition_this_batch;

      // Scheduling for multiple devices
      // Traditionally CcT takes a batch size (curr_B), a number of partitions (n_partition_this_batch),
      // and equally divides to obtain n_batch_per_partition_cpu. Then, 1 thread is launched per
      // partition with a batch size of n_batch_per_partition_cpu.
      // For CPU this works well. For GPU however, we want to have a single partition (i.e.
      // 1 thread, and 1 cuda stream for that thread), but with a larger batch size. However,
      // due to limited GPU memory, each physical operator should handle 1 image at a time.

      // If we want to schedule on the GPU, then we need to split the partitions differently
      const float proportion_of_images_on_GPU = layer_param->gpu_batch_proportion();
      // Determine how many of the CPU partitions to run on each device
      // We will split at granularity of cpu partitions
      // (e.g. a GPU will "take over" a number of those partitions from the CPU, into its own
      // single partition)
      const int num_CPU_partitions_in_GPU_partition = total_num_partitions_if_all_cpu * proportion_of_images_on_GPU;
      num_partitions_CPU = total_num_partitions_if_all_cpu - num_CPU_partitions_in_GPU_partition;
      n_batch_per_partition_gpu =  curr_B - n_batch_per_partition_cpu*num_partitions_CPU;
      if (num_CPU_partitions_in_GPU_partition > 0) {
        num_partitions_GPU = 1;
      }
      num_partitions = num_partitions_CPU + num_partitions_GPU;
    #ifndef _INCLUDE_GPUDRIVER
      if (num_CPU_partitions_in_GPU_partition > 0) {
        std::cout << "\nError, GPU not enabled. To enable running on GPU add NVCC and CUDA_INCLUDE to .config\n\n";
        assert(false);
      }
    #endif

      // std::cout << "  GPU Proportion     = " << layer_param->gpu_batch_proportion() << "\n";
      // std::cout << "  #Partitions Total  = " << num_partitions << "\n";
      // std::cout << "  #Partitions on CPU = " << num_partitions_CPU << "\n";
      // std::cout << "  #Partitions on GPU = " << num_partitions_GPU << "\n";
    }
  
    vector<LogicalCubeType *> _data_cubes_lower;
    vector<LogicalCubeType *> _grad_cubes_lower;

    vector<LogicalCubeType *> _data_cubes_higher;
    vector<LogicalCubeType *> _grad_cubes_higher;

    vector<LayerType *> _partitioned_layers_lower;
    vector<LayerType *> _partitioned_layers_higher;


    PhysicalStratum stratum;
};

#include "ParallelizedBridge_impl.hxx"

#endif
