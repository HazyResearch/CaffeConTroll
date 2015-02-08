//
//  ParallelizedConvolutionBridge.h
//  moka
//
//  Created by Firas Abuzaid on 2/8/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_ParallelizedConvolutionBridge_h
#define moka_ParallelizedConvolutionBridge_h

#include "AbstractBridge.h"
#include "PhysicalStratum.h"
#include "ConvolutionBridge.h"
#include <thread>
#include <vector>

// For now, we only support Layout_CRDB
template<typename DataType>
class ParallelizedConvolutionBridge : public AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB> {
  public:

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_forward_constructor;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_forward_last_transfer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_forward_history;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::run_with_n_threads;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_backward_updateweight_constructor;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_backward_updateweight_last_transfer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_backward_updateweight_history;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::layer_param;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::p_input_layer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::p_output_layer;

    typedef LogicalCube<DataType, Layout_CRDB> LogicalCubeType;

    typedef Layer<DataType, Layout_CRDB> LayerType;

    typedef ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, DataType,
            Layout_CRDB, DataType, Layout_CRDB> ConvolutionBridgeType;

    std::vector<LogicalCubeType *> _data_cubes_lower;
    std::vector<LogicalCubeType *> _grad_cubes_lower;

    std::vector<LogicalCubeType *> _data_cubes_higher;
    std::vector<LogicalCubeType *> _grad_cubes_higher;

    std::vector<LayerType *> _partitioned_layers_lower;
    std::vector<LayerType *> _partitioned_layers_higher;

    std::vector<ConvolutionBridgeType *> _bridges;

    PhysicalStratum stratum;

    const BridgeConfig * const config;

    const size_t n_partition;
    const size_t n_batch;
    const size_t n_thread_per_partition;
    const size_t n_batch_per_partition;

    // Network initialization constructor
    ParallelizedConvolutionBridge(Layer<DataType, Layout_CRDB> * const _input_layer,
        Layer<DataType, Layout_CRDB> * const _output_layer,
        const cnn::LayerParameter * const _layer_param, size_t _n_partition,
        size_t _n_thread_per_partition);

    // Testing constructor
    ParallelizedConvolutionBridge(Layer<DataType, Layout_CRDB> * const _input_layer,
        Layer<DataType, Layout_CRDB> * const _output_layer,
        const BridgeConfig * const _config, size_t _n_partition,
        size_t _n_thread_per_partition);

    void forward();

    void backward();

  private:
    void initialize();
};

#include "ParallelizedConvolutionBridge_impl.hxx"

#endif
