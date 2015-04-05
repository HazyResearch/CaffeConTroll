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
template<typename DataType, typename BridgeType, typename DriverClass>
class ParallelizedBridge : public AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass> {
  protected:
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::curr_B;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::input_d_cube;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::input_g_cube;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::output_d_cube;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::output_g_cube;

  public:
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_forward_constructor;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_forward_last_transfer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_forward_history;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::run_with_n_threads;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_backward_updateweight_constructor;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_backward_updateweight_last_transfer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_backward_updateweight_history;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::needs_to_calc_backward_grad;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::layer_param;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::solver_param;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_driver;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_input_layer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_output_layer;

    typedef LogicalCube<DataType, Layout_CRDB> LogicalCubeType;

    typedef Layer<DataType, Layout_CRDB> LayerType;

    // These are public for now, just so that we can write tests
    LogicalCubeType * p_model_cube; /** A ParallelizedConvolutionBridge should have a _single_
                                        copy of the model. Copy this model to different worker (or
                                        add optimization to share without copying) is the job
                                        of ParallelizedConvolutionBridge not its caller. **/
    LogicalCubeType * p_model_grad;
    LogicalCubeType * p_bias_grad;
    LogicalCubeType * p_bias_cube;

    const size_t n_partition;
    const size_t n_batch;
    const size_t n_thread_per_partition;
    const size_t n_batch_per_partition;

    float model_base_learning_rate;
    float bias_base_learning_rate;
    float model_base_regularization;
    float bias_base_regularization;

    GradientUpdater<DataType, DriverClass> * p_grad_updater;
    GradientUpdater<DataType, DriverClass> * p_grad_updater_bias;

    ParallelizedBridge(Layer<DataType, Layout_CRDB> * const _input_layer,
        Layer<DataType, Layout_CRDB> * const _output_layer,
        const cnn::LayerParameter * const _layer_param,
        const cnn::SolverParameter * const _solver_param,
        DriverClass * const _p_driver, size_t _n_partition,
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

    GradientUpdater<DataType, DriverClass> * const get_model_updater() {
        return p_grad_updater;
    }

    GradientUpdater<DataType, DriverClass> * const get_bias_updater() {
        return p_grad_updater_bias;
    }

  protected:
    vector<LogicalCubeType *> _data_cubes_lower;
    vector<LogicalCubeType *> _grad_cubes_lower;

    vector<LogicalCubeType *> _data_cubes_higher;
    vector<LogicalCubeType *> _grad_cubes_higher;

    vector<LayerType *> _partitioned_layers_lower;
    vector<LayerType *> _partitioned_layers_higher;

    vector<BridgeType *> _bridges;

    PhysicalStratum stratum;
};

#include "ParallelizedBridge_impl.hxx"

#endif
