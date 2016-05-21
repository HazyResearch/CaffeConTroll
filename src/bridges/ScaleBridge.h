//
//  ScaleBridge.h
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//
//  Description:  This bridge learns a scale factor and bias term for each channel

#ifndef _Scale_Bridge_h
#define _Scale_Bridge_h

#include "AbstractBridge.h"
#include "../util.h"

template
<typename InputLayerDataType, LayoutType InputLayerLayout,
  typename OutputLayerDataType, LayoutType OutputLayerLayout, typename DriverClass>
class ScaleBridge : public AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
  OutputLayerLayout, DriverClass> {
  public:
  
    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;
    typedef LogicalCube<InputLayerDataType, InputLayerLayout> LogicalCubeType;

    virtual void set_model_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * model) = 0;
    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * get_model_cube() = 0;
    virtual void set_bias_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * bias) = 0;
    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * get_bias_cube() = 0;

    ScaleBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param,
        const cnn::SolverParameter * const _solver_param,
        DriverClass * const _p_driver) {
      NOT_IMPLEMENTED;
    }
    
    ~ScaleBridge() {
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
class ScaleBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass>
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

    /* Re-declare these typedefs */
    typedef Layer<DataType, Layout_CRDB> InputLayerType;
    typedef Layer<DataType, Layout_CRDB> OutputLayerType;
    typedef LogicalCube<DataType, Layout_CRDB> LogicalCubeType;

    const bool bias_term;

    const cnn::FillerParameter weight_filler;
    const cnn::FillerParameter bias_filler;

    void set_model_cube(LogicalCube<DataType, Layout_CRDB> * model) {
#ifdef _DO_ASSERT
      // This is currently only ever be used as a special-case to prevent
      // a memcpy when copying from the host, so assert that this is true
      bool is_cpu_driver = std::is_same<DriverClass, CPUDriver>::value;
      assert(is_cpu_driver);
#endif
      p_model_cube->set_p_data(model->get_p_data());
    }
    
    LogicalCube<DataType, Layout_CRDB> * const get_model_cube() {
      return p_model_cube_shadow;
    }

    void set_bias_cube(LogicalCube<DataType, Layout_CRDB> * bias) {
#ifdef _DO_ASSERT
      assert(p_bias_cube);
      // This is currently only ever be used as a special-case to prevent
      // a memcpy when copying from the host, so assert that this is true
      bool is_cpu_driver = std::is_same<DriverClass, CPUDriver>::value;
      assert(is_cpu_driver);
#endif
      // SHADJIS TODO: Why this is a memcpy but set_model_cube wasn't?
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

    ScaleBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param,
        const cnn::SolverParameter * const _solver_param,
        DriverClass * const _p_driver);

    ~ScaleBridge();

    void forward();

    void backward();

  protected:
    LogicalCubeType * p_model_gradient_cube;
    LogicalCubeType * p_model_cube;
    LogicalCubeType * p_model_cube_shadow;

    LogicalCubeType * p_bias_gradient_cube;
    LogicalCubeType * p_bias_cube;

    LogicalCubeType * ones_bias_vector;

    void initialize_logical_cube(const LogicalCubeType * cube, const cnn::FillerParameter filler_param);

    int bias_param_id_;
    int axis_;
    int outer_dim_, scale_dim_, inner_dim_;
    LogicalCube<DataType, Layout_CRDB> * sum_multiplier_, * sum_result_, * temp_;


};

#include "ScaleBridge_impl.hxx"

#endif
