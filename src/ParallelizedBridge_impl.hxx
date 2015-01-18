//
//  PhysicalPlan_impl.hxx
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_PhysicalPlan_impl_hxx
#define moka_PhysicalPlan_impl_hxx

template<typename DATATYPE, LayoutType LAYOUTTYPE, BridgeType BRIDGETYPE, NonLinearFunction FUNC>
ParallelizedBridge<DATATYPE, LAYOUTTYPE, BRIDGETYPE, FUNC>::ParallelizedBridge(Layer<DATATYPE, Layout_CRDB> * const _layer_lower,
                       Layer<DATATYPE, Layout_CRDB> * const _layer_higher,
                       int _n_partition, int _n_thread_per_partition) :
    layer_lower(_layer_lower), layer_higher(_layer_higher),
    n_partition(_n_partition), n_batch(_layer_lower->dB),
    n_thread_per_partition(_n_thread_per_partition)
    {
        report_forward_constructor.reset();
        report_forward_last_transfer.reset();
        report_forward_history.reset();
        report_backward_updateweight_constructor.reset();
        report_backward_updateweight_last_transfer.reset();
        report_backward_updateweight_history.reset();


        /****
        *
        *   #####UGLY#####
        *   The following function needs to be refactored when the Physical-Logical
        *   refactoring is done.
        *
        ****/
        const int n_batch_per_partition = n_batch / n_partition;
        for(int b=0;b<n_batch;b+=n_batch_per_partition){

            const int n_batch_this_partition = b + n_batch_per_partition
            >= n_batch ? n_batch - b : n_batch_per_partition;

            _data_cubes_lower.push_back(
                                        new LogicalCubeType(layer_lower->p_data_cube->physical_get_RCDslice(b), layer_lower->dR, layer_lower->dC, layer_lower->dD, n_batch_this_partition)
                                        );

            _model_cubes_lower.push_back(
                                         layer_lower->p_model_cube
                                         );

            _grad_cubes_lower.push_back(
                                        new LogicalCubeType(layer_lower->p_gradient_cube->physical_get_RCDslice(b), layer_lower->gR, layer_lower->gC, layer_lower->gD, n_batch_this_partition)
                                        );


            _data_cubes_higher.push_back(
                                         new LogicalCubeType(layer_higher->p_data_cube->physical_get_RCDslice(b), layer_higher->dR, layer_higher->dC, layer_higher->dD, n_batch_this_partition)
                                         );

            _model_cubes_higher.push_back(
                                          layer_higher->p_model_cube
                                          );

            _grad_cubes_higher.push_back(
                                         new LogicalCubeType(layer_higher->p_gradient_cube->physical_get_RCDslice(b), layer_higher->gR, layer_higher->gC, layer_higher->gD, n_batch_this_partition)
                                         );

        }

        for(int ib=0;ib<_data_cubes_lower.size();ib++){
            _partitioned_layers_lower.push_back(
                                                new LayerType(_data_cubes_lower[ib], _model_cubes_lower[ib], _grad_cubes_lower[ib])
                                                );
            _partitioned_layers_higher.push_back(
                                                 new LayerType(_data_cubes_higher[ib], _model_cubes_higher[ib], _grad_cubes_higher[ib])
                                                 );
        }

        for(int ib=0;ib<_data_cubes_lower.size();ib++){
            _bridges.push_back(
                               new BridgeType(_partitioned_layers_lower[ib], _partitioned_layers_higher[ib])
                               );
        }

        for(int ib=0;ib<_data_cubes_lower.size();ib++){
            _bridges[ib]->run_with_n_threads = n_thread_per_partition;
            stratum.executors.push_back((PhysicalOperator*)_bridges[ib]);
        }

        report_backward_updateweight_constructor.end(0, 0, 0);
        report_forward_constructor.end(0, 0, 0);
    }

template<typename DATATYPE, LayoutType LAYOUTTYPE, BridgeType BRIDGETYPE, NonLinearFunction FUNC>
void ParallelizedBridge<DATATYPE, LAYOUTTYPE, BRIDGETYPE, FUNC>::forward(){
    report_forward_last_transfer.reset();
    stratum.forward();
    report_forward_last_transfer.end();
    report_forward_last_transfer.aggregate_onlystat(stratum.report_forward_last_transfer);
    report_forward_history.aggregate(report_forward_last_transfer);
}

template<typename DATATYPE, LayoutType LAYOUTTYPE, BridgeType BRIDGETYPE, NonLinearFunction FUNC>
void ParallelizedBridge<DATATYPE, LAYOUTTYPE, BRIDGETYPE, FUNC>::backward(){
    report_backward_updateweight_last_transfer.reset();
    stratum.backward();
    report_backward_updateweight_last_transfer.end();
    report_backward_updateweight_last_transfer.aggregate_onlystat(stratum.report_backward_updateweight_last_transfer);
    report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}



#endif
