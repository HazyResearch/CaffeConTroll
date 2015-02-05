//
//  PhysicalPlan.h
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_ParallelizedBridge_h
#define moka_ParallelizedBridge_h

#include "../PhysicalOperator.h"
#include "../PhysicalStratum.h"
#include <thread>
#include <vector>

template<typename DATATYPE, LayoutType LAYOUTTYPE, BridgeType BRIDGETYPE, NonLinearFunction FUNC>
class ParallelizedBridge : public PhysicalOperator {
public:

    typedef LogicalCube<DATATYPE, LAYOUTTYPE> LogicalCubeType;

    typedef Layer<DATATYPE, LAYOUTTYPE> LayerType;

    typedef Bridge<DATATYPE, LAYOUTTYPE, DATATYPE, LAYOUTTYPE, BRIDGETYPE, FUNC> BridgeType;

    LayerType * const layer_lower;
    LayerType * const layer_higher;

    std::vector<LogicalCubeType*> _data_cubes_lower;
    std::vector<LogicalCubeType*> _model_cubes_lower;
    std::vector<LogicalCubeType*> _grad_cubes_lower;

    std::vector<LogicalCubeType*> _data_cubes_higher;
    std::vector<LogicalCubeType*> _model_cubes_higher;
    std::vector<LogicalCubeType*> _grad_cubes_higher;

    std::vector<LayerType*> _partitioned_layers_lower;
    std::vector<LayerType*> _partitioned_layers_higher;

    std::vector<BridgeType*> _bridges;

    PhysicalStratum stratum;

    const int n_partition;
    const int n_batch;
    const int n_thread_per_partition;

    ParallelizedBridge(Layer<DATATYPE, Layout_CRDB> * const _layer_lower,
                       Layer<DATATYPE, Layout_CRDB> * const _layer_higher,
                       int _n_partition, int _n_thread_per_partition);

    void forward();

    void backward();

};

#include "ParallelizedBridge_impl.hxx"

#endif
