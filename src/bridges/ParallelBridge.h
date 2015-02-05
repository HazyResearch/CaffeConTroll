//
//  ParallelBridge.h
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_ParallelBridge_h
#define moka_ParallelBridge_h

#include "LogicalCube.h"
#include "Connector.h"
#include "Kernel.h"
#include "Report.h"
#include "Layer.h"
#include "Scanner.h"
#include "Bridge.h"

enum ParallelBridgeType {
    ParallelBridge_ParallelizeBy_DataBatch = 0,
    ParallelBridge_ParallelizeBy_ModelBatch = 1,
};

template
<typename InputLayerDataType, LayoutType InputLayerLayout,
typename OutputLayerDataType, LayoutType OutputLayerLayout,
BridgeType LAYERTYPE, NonLinearFunction FUNC>
class ParallelBridge{
public:

    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    const size_t i1R, i1C, i1D, i1B; /*< Size of the input LogicalCube 1 */
    const size_t i2R, i2C, i2D, i2B; /*< Size of the input LogicalCube 2 */
    const size_t oR, oC, oD, oB; /*< Size of the output LogicalCube */

    Report report_constructor;
    Report report_last_lowering;
    Report report_history;

    InputLayerType * const p_input_layer;
    OutputLayerType * const p_output_layer;

    Bridge(InputLayerType * const _p_input_layer,
           OutputLayerType * const _p_output_layer){
        std::cerr << "ERROR: Using a bridge with unsupported Layout or DataType." << std::endl;
        assert(false);
    }

    void forward(){
        std::cerr << "ERROR: Using a bridge with unsupported Layout or DataType." << std::endl;
        assert(false);
    }

    void backward(){
        std::cerr << "ERROR: Using a bridge with unsupported Layout or DataType." << std::endl;
        assert(false);
    }

};

#endif
