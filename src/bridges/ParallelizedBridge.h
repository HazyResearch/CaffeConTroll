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
    
    
    // Scheduler class members
    
    // A local CPU driver used by the scheduler
    // This is the same driver which templatizes the ParallelizedBridge,
    // and is used e.g. for collecting gradients
    CPUDriver * scheduler_local_cpudriver;
#ifdef _INCLUDE_GPUDRIVER
    // The GPU Driver, can add more drivers here to put into a vector
    GPUDriver * scheduler_gpudriver;
#else
    // SHADJIS TODO: If _INCLUDE_GPUDRIVER is undefined just make this a cpu driver
    // and assert it's never used. That way I don't have to ifdef it out everywhere.
    CPUDriver * scheduler_gpudriver;
#endif
    // Keep track of the number of partitions on the CPU and GPU
    size_t num_partitions_GPU;
    size_t num_partitions_CPU;
    
    // End of Scheduler class members
    

    const size_t n_partition;
    const size_t n_batch;
    const size_t n_thread_per_partition;
    const size_t n_batch_per_partition;

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
        size_t _n_thread_per_partition);

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
