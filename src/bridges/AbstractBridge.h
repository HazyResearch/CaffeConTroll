//
//  AbstractBridge.h
//  moka
//
//  Created by Firas Abuzaid on 1/22/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Abstract_Bridge_h
#define moka_Abstract_Bridge_h

#include "../LogicalCube.h"
#include "../Connector.h"
#include "../Kernel.h"
#include "../Report.h"
#include "../Scanner.h"
#include "PhysicalOperator.h"
#include "../Layer.h"
#include "../parser/cnn.pb.h"

template
<typename InputLayerDataType, LayoutType InputLayerLayout,
  typename OutputLayerDataType, LayoutType OutputLayerLayout>
class AbstractBridge : public PhysicalOperator {
  public:

    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    const size_t iR, iC, iD, iB; // Size of the input data, LogicalCube 1
    const size_t oR, oC, oD, oB; // Size of the output data, LogicalCube 2

    InputLayerType * const p_input_layer;
    OutputLayerType * const p_output_layer;

    const cnn::LayerParameter * const layer_param;

    Report report_constructor;
    Report report_last_lowering;
    Report report_history;

    void set_model_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * model) {}

    LogicalCube<InputLayerDataType, InputLayerLayout> * get_model_cube(){
        return NULL;
    }

    void set_bias_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * bias) {}    

    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * get_bias_cube() {
        return NULL;
    }

    // First constructor, which takes in a cnn::LayerParameter as a third argument. This will
    // be used when initializing from a *.prototxt file
    AbstractBridge<InputLayerDataType, InputLayerLayout,
      OutputLayerDataType, OutputLayerLayout>(InputLayerType * const _p_input_layer,
          OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param) :
        iR(_p_input_layer->p_data_cube->R), iC(_p_input_layer->p_data_cube->C), iD(_p_input_layer->p_data_cube->D),
        iB(_p_input_layer->p_data_cube->B), oR(_p_output_layer->p_data_cube->R), oC(_p_output_layer->p_data_cube->C),
        oD(_p_output_layer->p_data_cube->D), oB(_p_output_layer->p_data_cube->B),
        p_input_layer(_p_input_layer), p_output_layer(_p_output_layer), layer_param(_layer_param) {} // no-op, initialize only

    // Second constructor, which does NOT take in a cnn::LayerParameter as a third argument.
    // (Used primarily for testing)
    AbstractBridge<InputLayerDataType, InputLayerLayout,
      OutputLayerDataType, OutputLayerLayout>(InputLayerType * const _p_input_layer,
          OutputLayerType * const _p_output_layer) :
        iR(_p_input_layer->p_data_cube->R), iC(_p_input_layer->p_data_cube->C), iD(_p_input_layer->p_data_cube->D),
        iB(_p_input_layer->p_data_cube->B), oR(_p_output_layer->p_data_cube->R), oC(_p_output_layer->p_data_cube->C),
        oD(_p_output_layer->p_data_cube->D), oB(_p_output_layer->p_data_cube->B),
        p_input_layer(_p_input_layer), p_output_layer(_p_output_layer), layer_param(NULL) {} // no-op, initialize only
};

#endif
