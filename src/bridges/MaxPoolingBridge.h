//
//  MaxPoolingBridge.h
//  moka
//
//  Created by Firas Abuzaid on 1/22/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "../PhysicalOperator.h"
#include "AbstractBridge.h"
#include "../util.h"
#include "BridgeConfig.h"

#ifndef moka_Max_Pooling_Bridge_h
#define moka_Max_Pooling_Bridge_h
template
<typename InputLayerDataType, LayoutType InputLayerLayout, typename OutputLayerDataType, LayoutType OutputLayerLayout>
class MaxPoolingBridge : public AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType, OutputLayerLayout> {
  public:
    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    MaxPoolingBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
        const BridgeConfig * const _bconfig) {
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
class MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB> : public AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB> {
  public:
    /* Re-declare these member fields so that they don't have to be resolved using vtable lookups */
    using PhysicalOperator::report_forward_constructor;
    using PhysicalOperator::report_forward_last_transfer;
    using PhysicalOperator::report_forward_history;
    using PhysicalOperator::report_backward_updateweight_constructor;
    using PhysicalOperator::report_backward_updateweight_last_transfer;
    using PhysicalOperator::report_backward_updateweight_history;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::iR;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::iC;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::iD;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::iB;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oR;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oC;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oD;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oB;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::p_input_layer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::p_output_layer;

    /* Re-declare these typedefs */
    typedef Layer<DataType, Layout_CRDB> InputLayerType;
    typedef Layer<DataType, Layout_CRDB> OutputLayerType;

    MaxPoolingBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
        const BridgeConfig * const _bconfig);
    ~MaxPoolingBridge();

    const BridgeConfig * const bconfig;

    size_t pooled_height;
    size_t pooled_width;

    void forward();
    void backward();

  private:
    LogicalCube<size_t, Layout_CRDB> * max_index;
};

#include "MaxPoolingBridge_impl.hxx"

#endif
