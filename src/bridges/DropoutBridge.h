//
//  DropoutBridge.h
//
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Dropout_Bridge_h
#define moka_Dropout_Bridge_h

#include "AbstractBridge.h"
#include "../DeepNetConfig.h"
#include "../util.h"

template
<typename InputLayerDataType, LayoutType InputLayerLayout, typename OutputLayerDataType,
  LayoutType OutputLayerLayout, typename DriverClass>
class DropoutBridge : public AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
  OutputLayerLayout, DriverClass> {
  public:
    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    DropoutBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param, const cnn::SolverParameter * const _solver_param,
        DriverClass * const _p_driver) {
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
class DropoutBridge<DataType, Layout_CRDB, DataType, Layout_CRDB, DriverClass> : public AbstractBridge<DataType,
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

    /* Re-declare these typedefs */
    typedef Layer<DataType, Layout_CRDB> InputLayerType;
    typedef Layer<DataType, Layout_CRDB> OutputLayerType;

    DropoutBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
        const cnn::LayerParameter * const _layer_param, const cnn::SolverParameter * const _solver_param,
        DriverClass * const _p_driver);

    ~DropoutBridge();

    void forward();

    void backward();

    const float dropout_ratio;

    LogicalCube<unsigned int, Layout_CRDB> * mask_cube;

  protected:
    float scale;
    unsigned int uint_thres_;
    
    // SHADJIS TODO: This rng should not be part of this class but instead be part of the device driver 
    // (e.g. if this is a GPU bridge, no need for a CPU rng).
    // That requires calling p_driver->sbernoulli_initialize but for now I noticed that is 2ms slower 
    // for dropout6 and 7 fw (7ms -> 9ms), probably because of device memory pointers? Tested on g2.8xlarge
    std::random_device rd;
    mt19937 rng;
    boost::bernoulli_distribution<float> random_distribution;
    boost::variate_generator<mt19937, boost::bernoulli_distribution<float> > variate_generator;

};

#include "DropoutBridge_impl.hxx"

#endif
