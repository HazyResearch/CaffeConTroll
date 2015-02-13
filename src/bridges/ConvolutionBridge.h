//
//  ConvolutionBridge.h
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Convolution_Bridge_h
#define moka_Convolution_Bridge_h

#include "PhysicalOperator.h"
#include "AbstractBridge.h"
#include "../util.h"

enum ConvolutionBridgeType {
  CPU_CONV_LOWERINGTYPE1 = 0,
  CPU_CONV_LOWERINGTYPE2 = 1, // TODO: support the
  CPU_CONV_LOWERINGTYPE3 = 2  // other lowering types
};

/**
 * A ConvolutionBridge object connects two Layers, each of which is a (Data, Model, Gradient) triple.
 * A ConvolutionBridge contains two functions:
 *   - forward()
 *   - backward()
 * In the template,
 *   - LAYERTYPE defines {GPU, CPU} x {Types of lowering}
 *   - FUNC defines the non-linear function that will be applied to the output
 *     - {No func, TANH}
 *     - TODO: We need ReLU, etc.
 **/
template
<ConvolutionBridgeType LAYERTYPE, NonLinearFunction FUNC,
  typename InputLayerDataType, LayoutType InputLayerLayout,
  typename OutputLayerDataType, LayoutType OutputLayerLayout>
class ConvolutionBridge : public AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType, OutputLayerLayout> {
  public:

    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;
    typedef LogicalCube<InputLayerDataType, InputLayerLayout> LogicalCubeType;

    virtual void set_model_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * model) = 0;
    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * get_model_cube() = 0;
    virtual void set_bias_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * bias) = 0;
    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * get_bias_cube() = 0;

    // Testing constructor
    ConvolutionBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const BridgeConfig * const _config) {
      NOT_IMPLEMENTED;
    }

    // Network initialization constructor
    ConvolutionBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param) {
      NOT_IMPLEMENTED;
    }

    ~ConvolutionBridge() {
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
template <typename DataType, NonLinearFunction FUNC>
class ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB>
: public AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB> {
  public:
    /* Re-declare these member fields so that they don't have to be resolved using vtable lookups */
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_forward_constructor;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_forward_last_transfer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_forward_history;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::run_with_n_threads;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_backward_updateweight_constructor;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_backward_updateweight_last_transfer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::report_backward_updateweight_history;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::iR;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::iC;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::iD;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::iB;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oR;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oC;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oD;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::oB;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::layer_param;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::p_input_layer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::p_output_layer;

    Report report_forward_kernel;
    Report report_backward_kernel;
    Report report_forward_lowering;
    Report report_backward_inverse_lowering;
    Report report_backward_grad_kernel;
    Report report_backward_weight_kernel;

    /* Re-declare these typedefs */
    typedef Layer<DataType, Layout_CRDB> InputLayerType;
    typedef Layer<DataType, Layout_CRDB> OutputLayerType;
    typedef LogicalCube<DataType, Layout_CRDB> LogicalCubeType;

    const BridgeConfig * const config;

    const size_t K;
    const size_t num_output_features;
    const size_t stride;
    const size_t padding;

    const bool bias_term;
    const float stepsize;

    void set_model_cube(LogicalCube<DataType, Layout_CRDB> * model) {
        memcpy(p_model_cube->p_data, model->p_data, p_model_cube->n_elements*sizeof(DataType));
    }

    virtual LogicalCube<DataType, Layout_CRDB> * get_model_cube(){
        return model_cube();
    }

    void set_bias_cube(LogicalCube<DataType, Layout_CRDB> * bias) {
        memcpy(p_bias_cube->p_data, bias->p_data, p_bias_cube->n_elements*sizeof(DataType));        
    }    

    virtual LogicalCube<DataType, Layout_CRDB> * get_bias_cube() {
        return bias_cube();
    }

    // Testing constructor
    ConvolutionBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const BridgeConfig * const _config);

    // Network initialization constructor for convolution
    ConvolutionBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param);

    // Network initialization constructor
    // (Note: The presence of the 4th argument, inner_product,
    // let's us distinguish between this constructor and the
    // previous one. TODO: This is a terrible hack -- fix this!)
    ConvolutionBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param, const bool inner_product);

    ~ConvolutionBridge();

    void forward();

    void backward();

    LogicalCubeType * const model_cube();

    LogicalCubeType * const bias_cube();

  private:
    LogicalCubeType * p_model_cube;
    LogicalCubeType * p_bias_cube;
    LogicalCubeType * p_forward_lowered_data;

    cnn::FillerParameter weight_filler;
    cnn::FillerParameter bias_filler;

    size_t mR, mC, mD, mB; /*< Size of the model LogicalCube */

    Scanner<DataType, Layout_CRDB, FUNC> * p_forward_applyfunc_scanner;

    Connector<DataType, Layout_CRDB, DataType, Layout_CRDB, LOWERING_TYPE1>
      * p_forward_lower_connector;

    Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB,
      Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_NOTRANS> * p_forward_gemm_kernel;

    LogicalCube<DataType, Layout_CRDB> * p_backward_outputgrad;
    LogicalCube<DataType, Layout_CRDB> * p_backward_inputgrad;

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat,
      Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KernelConfig_TANHGRAD_ON_INPUT1> * p_backward_element_mul_kernel;

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat,
      Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_TRANS> * p_backward_gemm_updateweight_kernel;

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat,
      Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_TRANS_NOTRANS> * p_backward_gemm_updategrad_kernel;

    void initialize();

    void initialize_logical_cube(const LogicalCubeType * cube, const InitializerType initializer);
    void initialize_logical_cube(const LogicalCubeType * cube, const cnn::FillerParameter filler_param);
};

#include "ConvolutionBridge_impl.hxx"

#endif
