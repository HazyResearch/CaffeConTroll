//
//  SoftmaxLossBridge.h
//  moka
//
//  Created by Firas Abuzaid on 1/22/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Softmax_Loss_Bridge_h
#define moka_Softmax_Loss_Bridge_h

#include "AbstractBridge.h"
#include "../util.h"

template
<typename InputLayerDataType, LayoutType InputLayerLayout, typename OutputLayerDataType,
  LayoutType OutputLayerLayout, typename DriverClass>
class SoftmaxLossBridge : public AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
  OutputLayerLayout, DriverClass> {
  public:
    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;
    typedef LogicalCube<InputLayerDataType, InputLayerLayout> DataLabelsLogicalCubeType;

    SoftmaxLossBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
        DataLabelsLogicalCubeType * const _p_data_labels, DriverClass * const _p_driver) {
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
template <typename DataType, typename DriverClass>
class SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass> : public AbstractBridge<DataType, Layout_CRDB,
      DataType, Layout_CRDB, DriverClass> {
  protected:
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::curr_B;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::input_d_cube;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::input_g_cube;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::output_d_cube;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::output_g_cube;

  public:
    /* Re-declare these member fields so that they don't have to be resolved using vtable lookups */
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_forward_constructor;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_forward_last_transfer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_forward_history;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::run_with_n_threads;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_backward_updateweight_constructor;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_backward_updateweight_last_transfer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::report_backward_updateweight_history;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::iR;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::iC;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::iD;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::iB;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::oR;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::oC;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::oD;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::oB;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_input_layer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_output_layer;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_driver;

    /* Re-declare these typedefs */
    typedef Layer<DataType, Layout_CRDB> InputLayerType;
    typedef Layer<DataType, Layout_CRDB> OutputLayerType;
    typedef LogicalCube<DataType, Layout_CRDB> DataLabelsLogicalCubeType;

    // TODO: make this const again
    DataLabelsLogicalCubeType * const p_data_labels;

    const size_t ldR, ldC, ldD, ldB; /*< Size of the data labels LogicalCube */

    SoftmaxLossBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
        DataLabelsLogicalCubeType * const _p_data_labels, DriverClass * const _p_driver);

    void forward();

    void backward();

    DataType get_loss() {
      return loss;
    }

    void reset_loss() {
      loss = DataType(0.);
    }

  protected:
    DataType loss;
};

#include "SoftmaxLossBridge_impl.hxx"

#endif
