//
//  FullyConnectedBridge.h
//  moka
//
//  Created by Firas Abuzaid on 2/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_FullyConnected_Bridge_h
#define moka_FullyConnected_Bridge_h

#include "PhysicalOperator.h"
#include "AbstractBridge.h"
#include "../util.h"

/**
 * A FullyConnectedBridge implements the typical affine layer found in most CNNs.
 * TODO: There's WAY too much in common between this bridge and ConvolutionBridge;
 * we need to refactor the common code into a superclass.
 **/
template
<typename InputLayerDataType, LayoutType InputLayerLayout,
  typename OutputLayerDataType, LayoutType OutputLayerLayout, typename DriverClass>
class FullyConnectedBridge : public AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
  OutputLayerLayout, DriverClass> {
  public:

    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;
    typedef LogicalCube<InputLayerDataType, InputLayerLayout> LogicalCubeType;

    virtual void set_model_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * model) = 0;
    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * get_model_cube() = 0;
    virtual void set_bias_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * bias) = 0;
    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * get_bias_cube() = 0;

    FullyConnectedBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param,
        const cnn::SolverParameter * const _solver_param,
        DriverClass * const _p_driver) {
      NOT_IMPLEMENTED;
    }

    ~FullyConnectedBridge() {
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
class FullyConnectedBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>
: public AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass> {
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

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::layer_param;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::solver_param;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_driver;

    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_input_layer;
    using AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>::p_output_layer;

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

    const size_t K;
    const size_t num_output_features;
    const size_t stride;
    const size_t padding;

    const bool bias_term;

    const cnn::FillerParameter weight_filler;
    const cnn::FillerParameter bias_filler;

    void set_model_cube(LogicalCube<DataType, Layout_CRDB> * model) {
      p_model_cube->set_p_data(model->get_p_data());
    }

    LogicalCube<DataType, Layout_CRDB> * const get_model_cube() {
      return p_model_cube_shadow;
    }

    void set_bias_cube(LogicalCube<DataType, Layout_CRDB> * bias) {
      Util::_our_memcpy(p_bias_cube->get_p_data(), bias->get_p_data(), p_bias_cube->n_elements*sizeof(DataType));
    }

    LogicalCube<DataType, Layout_CRDB> * const get_bias_cube() {
      return p_bias_cube;
    }

    LogicalCube<DataType, Layout_CRDB> * const get_model_grad_cube() {
        return p_model_gradient_cube;
    }

    LogicalCube<DataType, Layout_CRDB> * const get_bias_grad_cube() {
        return p_bias_gradient_cube;
    }

    FullyConnectedBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param,
        const cnn::SolverParameter * const _solver_param,
        DriverClass * const _p_driver);

    ~FullyConnectedBridge();

    void forward();

    void backward();

  protected:
    LogicalCubeType * p_model_gradient_cube;
    LogicalCubeType * p_model_cube;
    LogicalCubeType * p_model_cube_shadow;

    LogicalCubeType * p_bias_gradient_cube;
    LogicalCubeType * p_bias_cube;

    LogicalCubeType * p_forward_lowered_data;

    size_t mR, mC, mD, mB; /*< Size of the model LogicalCube */

    Connector<DataType, Layout_CRDB, DataType, Layout_CRDB, LOWERING_TYPE1, DriverClass>
      * p_forward_lower_connector;

    Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB,
      Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_TRANS_NO_DIM_FLIP, DriverClass> * p_forward_gemm_kernel;

    LogicalCube<DataType, Layout_CRDB> * p_backward_inputgrad;

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat,
      Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_NOTRANS_NOTRANS_DIM_FLIP, DriverClass> * p_backward_gemm_updateweight_kernel;

    Kernel<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, DataType_SFFloat,
      Layout_CRDB, Kernel_GEMM_OpenBlas, KernelConfig_GEMM_TRANS_NOTRANS, DriverClass> * p_backward_gemm_updategrad_kernel;

    void initialize_logical_cube(const LogicalCubeType * cube, const cnn::FillerParameter filler_param);
};

#include "FullyConnectedBridge_impl.hxx"

#endif

