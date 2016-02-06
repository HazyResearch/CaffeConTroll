//
//  SplitBridge.h
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//
//  Description:  This bridge is the opposite of funnel. Whereas funnel is used when
//                a grouping of N merges into a grouping of 1 (in the forward direction),
//                split is used when a grouping of 1 splits into a grouping of N.
//

#ifndef _Split_Bridge_h
#define _Split_Bridge_h

#include "AbstractBridge.h"
#include "../util.h"

template
<typename InputLayerDataType, LayoutType InputLayerLayout, typename OutputLayerDataType,
  LayoutType OutputLayerLayout, typename DriverClass>
class SplitBridge : public AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
  OutputLayerLayout, DriverClass> {
  public:
    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    SplitBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param, const cnn::SolverParameter * const _solver_param) {
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
class SplitBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass> : public AbstractBridge<DataType,
      Layout_CRDB, DataType, Layout_CRDB, DriverClass> {
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

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::layer_param;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::solver_param;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_driver;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_input_layer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_output_layer;

    std::vector<Layer<DataType, Layout_CRDB>* > p_output_layers;

    /* Re-declare these typedefs */
    typedef Layer<DataType, Layout_CRDB> InputLayerType;
    typedef Layer<DataType, Layout_CRDB> OutputLayerType;

    SplitBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param, const cnn::SolverParameter * const _solver_param,
        DriverClass * const _p_driver);

    ~SplitBridge();

    void forward();

    void backward();
    
    void update_p_input_layer_data_CPU_ONLY(float * new_data)  {
      p_input_layer->p_data_cube->set_p_data(new_data);
      // For split bridge, also pass this pointer to all the output bridges
      for (size_t i = 0; i < p_output_layers.size(); ++i) {
        p_output_layers[i]->p_data_cube->set_p_data(new_data);
      }
    }
    
};

#include "SplitBridge_impl.hxx"

#endif
