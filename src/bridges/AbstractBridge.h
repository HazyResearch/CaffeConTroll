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
#include "../Layer.h"
#include "../Scanner.h"
#include "../PhysicalOperator.h"

template
<typename InputLayerDataType, LayoutType InputLayerLayout,
  typename OutputLayerDataType, LayoutType OutputLayerLayout>
class AbstractBridge : public PhysicalOperator {
  public:

    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    const size_t i1R, i1C, i1D, i1B; /*< Size of the input LogicalCube 1 */
    const size_t i2R, i2C, i2D, i2B; /*< Size of the input LogicalCube 2 */
    const size_t oR, oC, oD, oB; /*< Size of the output LogicalCube */

    InputLayerType * const p_input_layer;
    OutputLayerType * const p_output_layer;

    Report report_constructor;
    Report report_last_lowering;
    Report report_history;

    AbstractBridge<InputLayerDataType, InputLayerLayout,
      OutputLayerDataType, OutputLayerLayout>(InputLayerType * const _p_input_layer,
          OutputLayerType * const _p_output_layer)
        : i1R(_p_input_layer->p_data_cube->R),
        i1C(_p_input_layer->p_data_cube->C), i1D(_p_input_layer->p_data_cube->D),
        i1B(_p_input_layer->p_data_cube->B), i2R(_p_input_layer->p_model_cube->R),
        i2C(_p_input_layer->p_model_cube->C), i2D(_p_input_layer->p_model_cube->D),
        i2B(_p_input_layer->p_model_cube->B), oR(_p_output_layer->p_data_cube->R),
        oC(_p_output_layer->p_data_cube->C), oD(_p_output_layer->p_data_cube->D),
        oB(_p_output_layer->p_data_cube->B), p_input_layer(_p_input_layer),
        p_output_layer(_p_output_layer) {} // no-op, initialize only
};

#endif
