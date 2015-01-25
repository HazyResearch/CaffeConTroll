//
//  SoftmaxLossBridge.h
//  moka
//
//  Created by Firas Abuzaid on 1/22/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "../PhysicalOperator.h"
#include "AbstractBridge.h"
#include "../util.h"

#ifndef moka_Soft_Max_Loss_Bridge_h
#define moka_Soft_Max_Loss_Bridge_h
template
<typename InputLayerDataType, LayoutType InputLayerLayout, typename OutputLayerDataType, LayoutType OutputLayerLayout>
class SoftmaxLossBridge : public AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType, OutputLayerLayout> {
  public:
    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    SoftmaxLossBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer) {
      NOT_IMPLEMENTED;
    }

    void forward() {
      NOT_IMPLEMENTED;
    }

    void backward() {
      NOT_IMPLEMENTED;
    }
};

/******
 * Specializations
 */
template <typename DataType>
class SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB> : public AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB> {
  public:
    /* Re-declare these member fields so that they don't have to be resolved using vtable lookups */
    using PhysicalOperator::report_forward_constructor;
    using PhysicalOperator::report_forward_last_transfer;
    using PhysicalOperator::report_forward_history;
    using PhysicalOperator::run_with_n_threads;
    using PhysicalOperator::report_backward_updateweight_constructor;
    using PhysicalOperator::report_backward_updateweight_last_transfer;
    using PhysicalOperator::report_backward_updateweight_history;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::i1R;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::i1C;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::i1D;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::i1B;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::i2R;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::i2C;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::i2D;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::i2B;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oR;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oC;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oD;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oB;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::p_input_layer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::p_output_layer;

    /* Re-declare these typedefs */
    typedef Layer<DataType, Layout_CRDB> InputLayerType;
    typedef Layer<DataType, Layout_CRDB> OutputLayerType;

    SoftmaxLossBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer);

    void forward();
    void backward();
};

#include "SoftmaxLossBridge_impl.hxx"

#endif
