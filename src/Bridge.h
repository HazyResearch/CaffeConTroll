//
//  Layer.h
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "LogicalCube.h"
#include "Connector.h"
#include "Kernel.h"
#include "Report.h"
#include "Layer.h"
#include "Scanner.h"

#include "PhysicalOperator.h"

#ifndef moka_Bridge_h
#define moka_Bridge_h

enum BridgeType{
    Bridge_CPU_CONV_LOWERINGTYPE1 = 0
};

/**
 * A Bridge object connects two Layers, each of which is a (Data, Model, Gradient) triple.
 * A Bridge contains two functions:
 *   - forward()
 *   - backward()
 * In the template,
 *   - LAYERTYPE defines {GPU, CPU} x {Types of lowering}
 *   - FUNC defines the non-linear function that will be applied to the output
 *     - {No func, TANH}
 *     - TODO: We need ReLU, etc.
 **/
template
<typename InputLayerDataType, LayoutType InputLayerLayout,
typename OutputLayerDataType, LayoutType OutputLayerLayout,
BridgeType LAYERTYPE, NonLinearFunction FUNC>
class Bridge : public PhysicalOperator{
public:

    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    const size_t i1R, i1C, i1D, i1B; /*< Size of the input LogicalCube 1 */
    const size_t i2R, i2C, i2D, i2B; /*< Size of the input LogicalCube 2 */
    const size_t oR, oC, oD, oB; /*< Size of the output LogicalCube */

    InputLayerType * const p_input_layer;
    OutputLayerType * const p_output_layer;

    Report report_constructor;
    Report report_last_transfer;
    Report report_history;
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

/******
 * Specializations
 */
template<typename DataType, NonLinearFunction FUNC>
class Bridge<DataType, Layout_CRDB, DataType, Layout_CRDB, Bridge_CPU_CONV_LOWERINGTYPE1, FUNC> : public PhysicalOperator {
public:

    typedef Layer<DataType, Layout_CRDB> InputLayerType;
    typedef Layer<DataType, Layout_CRDB> OutputLayerType;

    const size_t i1R, i1C, i1D, i1B; /*< Size of the input LogicalCube 1 */
    const size_t i2R, i2C, i2D, i2B; /*< Size of the input LogicalCube 2 */
    const size_t oR, oC, oD, oB; /*< Size of the output LogicalCube */

    InputLayerType * const p_input_layer;
    OutputLayerType * const p_output_layer;

    float stepsize;

    Scanner<DataType, Layout_CRDB, FUNC> * p_forward_applyfunc_scanner;

    Connector<DataType, Layout_CRDB, DataType, Layout_CRDB, LOWERING_TYPE1> *
    p_forward_lower_connector;

    LogicalCube<DataType, Layout_CRDB> * p_forward_lowered_data;

    LoweringConfig lconfig_forward;

    Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_NOTRANS> * p_forward_gemm_kernel;

    LogicalCube<DataType, Layout_CRDB> * p_backward_outputgrad;
    LogicalCube<DataType, Layout_CRDB> * p_backward_inputgrad;

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KernelConfig_TANHGRAD_ON_INPUT1> * p_backward_element_mul_kernel;

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_TRANS> * p_backward_gemm_updateweight_kernel;

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_TRANS_NOTRANS> * p_backward_gemm_updategrad_kernel;

    Bridge(InputLayerType * const _p_input_layer,
           OutputLayerType * const _p_output_layer);

    void forward();

    void backward();

};

#include "Bridge_impl.hxx"

#endif






