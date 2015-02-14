//
//  GroupConvolutionBridge.h
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_GroupConvolution_Bridge_h
#define moka_GroupConvolution_Bridge_h

#include "PhysicalOperator.h"
#include "AbstractBridge.h"
#include "ConvolutionBridge.h"
#include "../util.h"
#include <vector>

using std::vector;

/**
 * Same as the ConvolutionBridge, but handles grouping > 1
 **/
template
<ConvolutionBridgeType LAYERTYPE, NonLinearFunction FUNC,
  typename InputLayerDataType, LayoutType InputLayerLayout,
  typename OutputLayerDataType, LayoutType OutputLayerLayout>
class GroupConvolutionBridge : public AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType, OutputLayerLayout> {
  public:

    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;
    typedef LogicalCube<InputLayerDataType, InputLayerLayout> LogicalCubeType;

    virtual void set_model_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * model) = 0;
    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * get_model_cube() = 0;
    virtual void set_bias_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * bias) = 0;
    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * get_bias_cube() = 0;

    GroupConvolutionBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param) {
      NOT_IMPLEMENTED;
    }

    ~GroupConvolutionBridge() {
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
class GroupConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB>
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

    /* Re-declare these typedefs */
    typedef Layer<DataType, Layout_CRDB> InputLayerType;
    typedef Layer<DataType, Layout_CRDB> OutputLayerType;
    typedef LogicalCube<DataType, Layout_CRDB> LogicalCubeType;

    typedef ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC, DataType, Layout_CRDB, DataType, Layout_CRDB> ConvBridge;

    const size_t K;
    const size_t num_output_features;
    const size_t stride;
    const size_t padding;

    const bool bias_term;

    const cnn::FillerParameter weight_filler;
    const cnn::FillerParameter bias_filler;

    const size_t grouping;

    void set_model_cube(LogicalCube<DataType, Layout_CRDB> * model) {
      Util::_our_memcpy(p_model_cube->p_data, model->p_data, p_model_cube->n_elements*sizeof(DataType));
    }

    LogicalCube<DataType, Layout_CRDB> * const get_model_cube() {
      return p_model_cube;
    }

    void set_bias_cube(LogicalCube<DataType, Layout_CRDB> * bias) {
      Util::_our_memcpy(p_bias_cube->p_data, bias->p_data, p_bias_cube->n_elements*sizeof(DataType));
    }

    LogicalCube<DataType, Layout_CRDB> * const get_bias_cube() {
      return p_bias_cube;
    }

    GroupConvolutionBridge(InputLayerType * const _p_input_layer,
        OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param);

    ~GroupConvolutionBridge();

    void forward();

    void backward();

  protected:
    LogicalCubeType * p_model_cube;
    LogicalCubeType * p_bias_cube;

    vector<ConvBridge *> conv_bridges;
    vector<cnn::LayerParameter *> layer_params;

    vector<LogicalCubeType *> input_data_cubes;
    vector<LogicalCubeType *> input_grad_cubes;
    vector<LogicalCubeType *> output_data_cubes;
    vector<LogicalCubeType *> output_grad_cubes;

    vector<InputLayerType *> input_layers;
    vector<InputLayerType *> output_layers;
};

#include "GroupConvolutionBridge_impl.hxx"

#endif
