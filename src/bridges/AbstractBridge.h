//
//  AbstractBridge.h
//
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
#include "../algorithms/GradientUpdater.h"

#include "../sched/DeviceDriver_CPU.h"
#ifdef _INCLUDE_GPUDRIVER
#include "../sched/DeviceDriver_GPU.h"
#endif

template
<typename InputLayerDataType, LayoutType InputLayerLayout,
  typename OutputLayerDataType, LayoutType OutputLayerLayout, typename DriverClass>
class AbstractBridge : public PhysicalOperator {
  protected:
    // SHADJIS TODO: curr_B is only used in parallelized bridge now,
    // but I think it should be used everywhere to handle the last batch
    size_t curr_B;
    LogicalCube<InputLayerDataType, InputLayerLayout> * input_d_cube;
    LogicalCube<InputLayerDataType, InputLayerLayout> * input_g_cube;
    LogicalCube<OutputLayerDataType, OutputLayerLayout> * output_d_cube;
    LogicalCube<OutputLayerDataType, OutputLayerLayout> * output_g_cube;

  public:
    std::string name; // lets give Bridge a name

    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    const size_t iR, iC, iD, iB; // Size of the input data, LogicalCube 1
    const size_t oR, oC, oD, oB; // Size of the output data, LogicalCube 2

    InputLayerType * const p_input_layer;
    OutputLayerType * const p_output_layer;

    const cnn::LayerParameter * const layer_param;
    const cnn::SolverParameter * const solver_param;

    DriverClass * const p_driver;

    bool needs_to_calc_backward_grad;

    Report report_constructor;
    Report report_last_lowering;
    Report report_history;

    bool bias_term;

    void report_forward() {
        std::cout << std::endl;
        std::cout << "## FORWARD REPORT OF LAYER " << name << " ##" << std::endl;
        report_forward_last_transfer.print();
    }

    void report_backward() {
        std::cout << std::endl;
        std::cout << "## BACKWARD REPORT OF LAYER " << name << " ##" << std::endl;
        report_backward_updateweight_last_transfer.print();
    }

    // SHADJIS TODO: I'm not sure if we ever need this. I think this should always just
    // be handled with a direct driver->memcpy(), and if the src/dst are host or device
    // is abstracted to the caller
    void copy_from_host_to_device(LogicalCube<InputLayerDataType, InputLayerLayout> * const dst,
	LogicalCube<InputLayerDataType, InputLayerLayout> * const src) {
        // We know local is a CPU driver
        CPUDriver *local_cpu_driver = new CPUDriver();
        p_driver->memcpy(dst->get_device_pointer(p_driver), src->get_device_pointer(local_cpu_driver));
        delete local_cpu_driver;
    }

    // SHADJIS TODO: See comment above for copy_from_host_to_device, this seems unnecessary
    void copy_from_device_to_host(LogicalCube<InputLayerDataType, InputLayerLayout> * const dst,
	LogicalCube<InputLayerDataType, InputLayerLayout> * const src) {
        // We know local is a CPU driver
        CPUDriver *local_cpu_driver = new CPUDriver();
        p_driver->memcpy(dst->get_device_pointer(local_cpu_driver), src->get_device_pointer(p_driver));
        delete local_cpu_driver;
    }

    // Bridges which subclass AbstractBridge may override these four methods later
    // (e.g. ConvolutionBridge). Most, however, won't, since only ConvolutionBridge
    // and FullyConnected Bridge have weights that need to be updated
    virtual void set_model_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * model) {}

    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * const get_model_cube() {
        return NULL;
    }

    virtual void set_bias_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * bias) {}

    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * const get_bias_cube() {
        return NULL;
    }

    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * const get_model_grad_cube() {
        return NULL;
    }

    LogicalCube<InputLayerDataType, InputLayerLayout> * const get_bias_grad_cube() {
        return NULL;
    }

    // Need these for snapshot tests
    virtual GradientUpdater<InputLayerDataType, CPUDriver> * const get_model_updater() {
        return NULL;
    }

    virtual GradientUpdater<InputLayerDataType, CPUDriver> * const get_bias_updater() {
        return NULL;
    }

    virtual void set_curr_batch_size(const size_t _curr_B) {
      curr_B = _curr_B;
    }

    // First constructor, which takes in a cnn::LayerParameter as a third argument. This will
    // be used when initializing from a *.prototxt file
    AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
      OutputLayerLayout, DriverClass>(InputLayerType * const _p_input_layer,
          OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
          const cnn::SolverParameter * const _solver_param, DriverClass * const _p_driver) :
        curr_B(_p_input_layer->p_data_cube->B), iR(_p_input_layer->p_data_cube->R),
        iC(_p_input_layer->p_data_cube->C), iD(_p_input_layer->p_data_cube->D),
        iB(_p_input_layer->p_data_cube->B), oR(_p_output_layer->p_data_cube->R),
        oC(_p_output_layer->p_data_cube->C), oD(_p_output_layer->p_data_cube->D),
        oB(_p_output_layer->p_data_cube->B), p_input_layer(_p_input_layer),
        p_output_layer(_p_output_layer), layer_param(_layer_param),
        solver_param(_solver_param), p_driver(_p_driver), bias_term(false) {

          // Default non-softmax: Use constructor to own data. Allocates on the device.
          assert(false); // For now nothing should use this default constructor, change this for new devices
          input_d_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(iR, iC, iD, iB, p_driver);
          input_g_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(iR, iC, iD, iB, p_driver);
          output_d_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(oR, oC, oD, oB, p_driver);
          output_g_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(oR, oC, oD, oB, p_driver);
        }

    // Second constructor, which does NOT take in a cnn::LayerParameter as a third argument.
    // (Used only for Softmax)
    AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
      OutputLayerLayout, DriverClass>(InputLayerType * const _p_input_layer,
          OutputLayerType * const _p_output_layer, DriverClass * const _p_driver) :
        curr_B(_p_input_layer->p_data_cube->B),
        iR(_p_input_layer->p_data_cube->R), iC(_p_input_layer->p_data_cube->C),
        iD(_p_input_layer->p_data_cube->D), iB(_p_input_layer->p_data_cube->B),
        oR(_p_output_layer->p_data_cube->R), oC(_p_output_layer->p_data_cube->C),
        oD(_p_output_layer->p_data_cube->D), oB(_p_output_layer->p_data_cube->B),
        p_input_layer(_p_input_layer), p_output_layer(_p_output_layer),
        layer_param(NULL), solver_param(NULL), p_driver(_p_driver),
        bias_term(false) {

          // Default softmax: Use constructor to own data. Allocates on the device.
          assert(false); // For now nothing should use this default constructor, change this for new devices
          input_d_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(iR, iC, iD, iB, p_driver);
          input_g_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(iR, iC, iD, iB, p_driver);
          output_d_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(oR, oC, oD, oB, p_driver);
          output_g_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(oR, oC, oD, oB, p_driver);
        }

    // This needs to be virtual, so we can delete the subclass bridges
    virtual ~AbstractBridge() {
      delete input_d_cube;  input_d_cube  = NULL;
      delete input_g_cube;  input_g_cube  = NULL;
      delete output_d_cube; output_d_cube = NULL;
      delete output_g_cube; output_g_cube = NULL;
    }

    // Update the pointer of the input layer's data
    // This function is used to run a different dataset on the network
    // (e.g. for validation set)
    // SHADJIS TODO: For now this is only needed for the first layer
    // If we want to do this for other layers, need to copy to GPU for
    // GPU bridges. For now these functions assume pointers are host pointers.
    // More generally they should be device memory pointers.
    // For now, just assume on CPU (host). In the future we might want to do direct
    // copy of data to GPU memory, then this function will get a device memory pointer.
    virtual void update_p_input_layer_data_CPU_ONLY(InputLayerDataType * new_data)  {
      // SHADJIS TODO: Assert this bridge does not share input layer with prev bridge
      // and both are on device, i.e. assert that data is on host
      p_input_layer->p_data_cube->set_p_data(new_data);
    }
    // Update the pointer of the output layer's gradient
    // This function is used to update the gradients of this layer which
    // come from another source, e.g. a separate network
    // SHADJIS TODO: As above, this is only for layers on the CPU.
    // If this bridge is a GPU bridge and it shares its output layer with the next 
    // bridge's input layer then the grad data will not be copied back to the host so
    // the pointer passed in here needs to be a device pointer. More generally therefore
    // a device memory pointer should be passed into this function.
    // For now, just assume on CPU (host). In the future we might want to do direct
    // copy of gradients back to GPU memory, then this function will get a device memory pointer.
    virtual void update_p_output_layer_gradient_CPU_ONLY(InputLayerDataType * new_data)  {
      // SHADJIS TODO: Assert this bridge does not share output layer with next bridge,
      // and both are on device, i.e. assert that grad is on host
      p_output_layer->p_gradient_cube->set_p_data(new_data);
    }


    // SHADJIS TODO: These functions aren't needed, all these members are public.
    // Should just call these directly rather than add new functions.
    // Get sizes
    virtual size_t get_input_data_size()  {
      return input_d_cube->n_elements;
    }
    virtual size_t get_output_data_size()  {
      return output_d_cube->n_elements;
    }
};

// CPUDriver specialization
template
<typename InputLayerDataType, LayoutType InputLayerLayout,
  typename OutputLayerDataType, LayoutType OutputLayerLayout>
class AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
      OutputLayerLayout, CPUDriver> : public PhysicalOperator {
  protected:
    // the size of the current training batch
    // always <= iB
    size_t curr_B;
    LogicalCube<InputLayerDataType, InputLayerLayout> * input_d_cube;
    LogicalCube<InputLayerDataType, InputLayerLayout> * input_g_cube;
    LogicalCube<OutputLayerDataType, OutputLayerLayout> * output_d_cube;
    LogicalCube<OutputLayerDataType, OutputLayerLayout> * output_g_cube;

  public:
    std::string name; // lets give Bridge a name

    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    const size_t iR, iC, iD, iB; // Size of the input data, LogicalCube 1
    const size_t oR, oC, oD, oB; // Size of the output data, LogicalCube 2

    InputLayerType * const p_input_layer;
    OutputLayerType * const p_output_layer;

    const cnn::LayerParameter * const layer_param;
    const cnn::SolverParameter * const solver_param;

    CPUDriver * const p_driver;

    bool needs_to_calc_backward_grad;

    Report report_constructor;
    Report report_last_lowering;
    Report report_history;

    bool bias_term;

    void report_forward() {
        std::cout << std::endl;
        std::cout << "## FORWARD REPORT OF LAYER " << name << " ##" << std::endl;
        report_forward_last_transfer.print();
    }

    void report_backward() {
        std::cout << std::endl;
        std::cout << "## BACKWARD REPORT OF LAYER " << name << " ##" << std::endl;
        report_backward_updateweight_last_transfer.print();
    }

    // SHADJIS TODO: I'm not sure if we ever need this. I think this should always just
    // be handled with a direct driver->memcpy(), and if the src/dst are host or device
    // is abstracted to the caller
    // If p_driver == CPUDriver, then we just need to reassign pointers
    void copy_from_host_to_device(LogicalCube<InputLayerDataType, InputLayerLayout> * const dst,
	LogicalCube<InputLayerDataType, InputLayerLayout> * const src) {
      dst->set_p_data(src->get_p_data());
    }

    // SHADJIS TODO: See comment above for copy_from_host_to_device, this seems unnecessary
    void copy_from_device_to_host(LogicalCube<InputLayerDataType, InputLayerLayout> * const dst,
	LogicalCube<InputLayerDataType, InputLayerLayout> * const src) {
      dst->set_p_data(src->get_p_data());
    }

    // Bridges which subclass AbstractBridge may override these four methods later
    // (e.g. ConvolutionBridge). Most, however, won't, since only ConvolutionBridge
    // and FullyConnected Bridge have weights that need to be updated
    virtual void set_model_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * model) {}

    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * const get_model_cube() {
        return NULL;
    }

    virtual void set_bias_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * bias) {}

    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * const get_bias_cube() {
        return NULL;
    }

    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * const get_model_grad_cube() {
        return NULL;
    }

    LogicalCube<InputLayerDataType, InputLayerLayout> * const get_bias_grad_cube() {
        return NULL;
    }

    // Need these for snapshot tests
    virtual GradientUpdater<InputLayerDataType, CPUDriver> * const get_model_updater() {
        return NULL;
    }

    virtual GradientUpdater<InputLayerDataType, CPUDriver> * const get_bias_updater() {
        return NULL;
    }

    void set_curr_batch_size(const size_t _curr_B) {
      curr_B = _curr_B;
    }

    // First constructor, which takes in a cnn::LayerParameter as a third argument. This will
    // be used when initializing from a *.prototxt file
    AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
      OutputLayerLayout, CPUDriver>(InputLayerType * const _p_input_layer,
          OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
          const cnn::SolverParameter * const _solver_param, CPUDriver * const _p_driver) :
        curr_B(_p_input_layer->p_data_cube->B), iR(_p_input_layer->p_data_cube->R),
        iC(_p_input_layer->p_data_cube->C), iD(_p_input_layer->p_data_cube->D),
        iB(_p_input_layer->p_data_cube->B), oR(_p_output_layer->p_data_cube->R),
        oC(_p_output_layer->p_data_cube->C), oD(_p_output_layer->p_data_cube->D),
        oB(_p_output_layer->p_data_cube->B), p_input_layer(_p_input_layer),
        p_output_layer(_p_output_layer), layer_param(_layer_param),
        solver_param(_solver_param), p_driver(_p_driver), bias_term(false) {

          // CPU: Use constructor to not own data. Does not allocate on the device.
          input_d_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(NULL, iR, iC, iD, iB, p_driver);
          input_g_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(NULL, iR, iC, iD, iB, p_driver);
          output_d_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(NULL, oR, oC, oD, oB, p_driver);
          output_g_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(NULL, oR, oC, oD, oB, p_driver);
        }

    // Second constructor, which does NOT take in a cnn::LayerParameter as a third argument.
    // (Used only for Softmax)
    AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
      OutputLayerLayout, CPUDriver>(InputLayerType * const _p_input_layer,
          OutputLayerType * const _p_output_layer, CPUDriver * const _p_driver) :
        curr_B(_p_input_layer->p_data_cube->B),
        iR(_p_input_layer->p_data_cube->R), iC(_p_input_layer->p_data_cube->C),
        iD(_p_input_layer->p_data_cube->D), iB(_p_input_layer->p_data_cube->B),
        oR(_p_output_layer->p_data_cube->R), oC(_p_output_layer->p_data_cube->C),
        oD(_p_output_layer->p_data_cube->D), oB(_p_output_layer->p_data_cube->B),
        p_input_layer(_p_input_layer), p_output_layer(_p_output_layer),
        layer_param(NULL), solver_param(NULL), p_driver(_p_driver),
        bias_term(false) {

          // CPU: Use constructor to not own data. Does not allocate on the device.
          input_d_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(NULL, iR, iC, iD, iB, p_driver);
          input_g_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(NULL, iR, iC, iD, iB, p_driver);
          output_d_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(NULL, oR, oC, oD, oB, p_driver);
          output_g_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(NULL, oR, oC, oD, oB, p_driver);
        }

    // This needs to be virtual, so we can delete the subclass bridges
    virtual ~AbstractBridge() {
      delete input_d_cube;  input_d_cube  = NULL;
      delete input_g_cube;  input_g_cube  = NULL;
      delete output_d_cube; output_d_cube = NULL;
      delete output_g_cube; output_g_cube = NULL;
    }
    
    // SHADJIS TODO:
    // For the CPU abstract bridge only I will also declare some functions which are only 
    // related to the scheduler, and therefore only called by parallelized bridge (only
    // relevant for abstract bridge with cpu driver since pbridge always uses gpu driver).
    // The reason I am adding these here now is because I want to call these for pbridges,
    // but all I have in DeepNet is a vector of abstractbridges (i.e. some are pbridgrs, some
    // are softmax/dropout/funnel/etc.). Morever, even if all bridges were created by pbridges,
    // I would still have a vector of all different types of pbridges (since pbridge currently
    // has the sub-bridge type as a template). So to avoid dealing with templates I am putting
    // these scheduler functions here, and overloading them in the pbridge (which handles 
    // scheduling within layers). Should fix the abstraction so this is not part of abstract
    // bridge, but only parallelized bridge, or e.g. some scheduler class.
    virtual void set_share_pointer_with_prev_bridge(bool _share) { assert(false); }
    virtual bool get_share_pointer_with_prev_bridge() { assert(false); }
    virtual void set_share_pointer_with_next_bridge(bool _share) { assert(false); }
    virtual bool get_share_pointer_with_next_bridge() { assert(false); }
    virtual size_t get_num_partitions_CPU() { assert(false); }
    virtual std::vector <size_t> get_GPU_batch_sizes() { assert(false); }
    virtual std::vector <int> get_used_gpu_to_device_id_map() { assert(false); }
    virtual std::vector< LogicalCube<InputLayerDataType, Layout_CRDB> *> get_data_cubes_higher() { assert(false); }
    virtual std::vector< LogicalCube<InputLayerDataType, Layout_CRDB> *> get_grad_cubes_higher() { assert(false); }
    virtual void force_host_to_device_model_copy() { assert(false); }
    virtual void force_host_to_device_bias_copy()  { assert(false); }
    virtual void force_device_to_host_model_copy() { assert(false); }
    virtual void force_device_to_host_bias_copy()  { assert(false); }
    virtual InputLayerDataType * get_model_gradient_host() { assert(false); }
    virtual InputLayerDataType * get_bias_gradient_host()  { assert(false); }
    virtual void update_model_with_gradient_CPU(InputLayerDataType * grad)  { assert(false); }
    virtual void update_bias_with_gradient_CPU(InputLayerDataType * grad)   { assert(false); }
    virtual void set_update_model_gradients(bool _update_model_gradients) {}

    // Model parallelism PBridge members
    virtual int get_model_parallelism_group_size() { return 1; }
    virtual void set_model_parallelism_group_size(int _model_parallelism_group_size) { assert(false); }

    // Update the pointer of the input layer's data
    // This function is used to run a different dataset on the network
    // (e.g. for validation set)
    virtual void update_p_input_layer_data_CPU_ONLY(InputLayerDataType * new_data)  {
      p_input_layer->p_data_cube->set_p_data(new_data);
    }
    // Update the pointer of the output layer's gradient
    // This function is used to update the gradients of this layer which
    // come from another source, e.g. a separate network
    virtual void update_p_output_layer_gradient_CPU_ONLY(InputLayerDataType * new_data)  {
      p_output_layer->p_gradient_cube->set_p_data(new_data);
    }
    
    // Get sizes
    virtual size_t get_input_data_size()  {
      return input_d_cube->n_elements;
    }
    virtual size_t get_output_data_size()  {
      return output_d_cube->n_elements;
    }
};


#ifdef _INCLUDE_GPUDRIVER
// GPUDriver specialization
template
<typename InputLayerDataType, LayoutType InputLayerLayout,
  typename OutputLayerDataType, LayoutType OutputLayerLayout>
class AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
      OutputLayerLayout, GPUDriver> : public PhysicalOperator {
  protected:
    // the size of the current training batch
    // always <= iB
    size_t curr_B;
    LogicalCube<InputLayerDataType, InputLayerLayout> * input_d_cube;
    LogicalCube<InputLayerDataType, InputLayerLayout> * input_g_cube;
    LogicalCube<OutputLayerDataType, OutputLayerLayout> * output_d_cube;
    LogicalCube<OutputLayerDataType, OutputLayerLayout> * output_g_cube;

  public:
    std::string name; // lets give Bridge a name

    typedef Layer<InputLayerDataType, InputLayerLayout> InputLayerType;
    typedef Layer<OutputLayerDataType, OutputLayerLayout> OutputLayerType;

    const size_t iR, iC, iD, iB; // Size of the input data, LogicalCube 1
    const size_t oR, oC, oD, oB; // Size of the output data, LogicalCube 2

    InputLayerType * const p_input_layer;
    OutputLayerType * const p_output_layer;

    const cnn::LayerParameter * const layer_param;
    const cnn::SolverParameter * const solver_param;

    GPUDriver * const p_driver;

    bool needs_to_calc_backward_grad;

    Report report_constructor;
    Report report_last_lowering;
    Report report_history;

    bool bias_term;

    void report_forward() {
        std::cout << std::endl;
        std::cout << "## FORWARD REPORT OF LAYER " << name << " ##" << std::endl;
        report_forward_last_transfer.print();
    }

    void report_backward() {
        std::cout << std::endl;
        std::cout << "## BACKWARD REPORT OF LAYER " << name << " ##" << std::endl;
        report_backward_updateweight_last_transfer.print();
    }

    // SHADJIS TODO: I'm not sure if we ever need this. I think this should always just
    // be handled with a direct driver->memcpy(), and if the src/dst are host or device
    // is abstracted to the caller
    void copy_from_host_to_device(LogicalCube<InputLayerDataType, InputLayerLayout> * const dst,
	LogicalCube<InputLayerDataType, InputLayerLayout> * const src) {
        // We know local is a CPU driver
        CPUDriver *local_cpu_driver = new CPUDriver();
        p_driver->memcpy(dst->get_device_pointer(p_driver), src->get_device_pointer(local_cpu_driver));
        delete local_cpu_driver;
    }

    // SHADJIS TODO: See comment above for copy_from_host_to_device, this seems unnecessary
    void copy_from_device_to_host(LogicalCube<InputLayerDataType, InputLayerLayout> * const dst,
	LogicalCube<InputLayerDataType, InputLayerLayout> * const src) {
        // We know local is a CPU driver
        CPUDriver *local_cpu_driver = new CPUDriver();
        p_driver->memcpy(dst->get_device_pointer(local_cpu_driver), src->get_device_pointer(p_driver));
        delete local_cpu_driver;
    }

    // Bridges which subclass AbstractBridge may override these four methods later
    // (e.g. ConvolutionBridge). Most, however, won't, since only ConvolutionBridge
    // and FullyConnected Bridge have weights that need to be updated
    virtual void set_model_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * model) {}

    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * const get_model_cube() {
        return NULL;
    }

    virtual void set_bias_cube(LogicalCube<InputLayerDataType, InputLayerLayout> * bias) {}

    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * const get_bias_cube() {
        return NULL;
    }

    virtual LogicalCube<InputLayerDataType, InputLayerLayout> * const get_model_grad_cube() {
        return NULL;
    }

    LogicalCube<InputLayerDataType, InputLayerLayout> * const get_bias_grad_cube() {
        return NULL;
    }

    // Need these for snapshot tests
    virtual GradientUpdater<InputLayerDataType, GPUDriver> * const get_model_updater() {
        return NULL;
    }

    virtual GradientUpdater<InputLayerDataType, GPUDriver> * const get_bias_updater() {
        return NULL;
    }

    void set_curr_batch_size(const size_t _curr_B) {
      curr_B = _curr_B;
    }

    // First constructor, which takes in a cnn::LayerParameter as a third argument. This will
    // be used when initializing from a *.prototxt file
    AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
      OutputLayerLayout, GPUDriver>(InputLayerType * const _p_input_layer,
          OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param,
          const cnn::SolverParameter * const _solver_param, GPUDriver * const _p_driver) :
        curr_B(_p_input_layer->p_data_cube->B), iR(_p_input_layer->p_data_cube->R),
        iC(_p_input_layer->p_data_cube->C), iD(_p_input_layer->p_data_cube->D),
        iB(_p_input_layer->p_data_cube->B), oR(_p_output_layer->p_data_cube->R),
        oC(_p_output_layer->p_data_cube->C), oD(_p_output_layer->p_data_cube->D),
        oB(_p_output_layer->p_data_cube->B), p_input_layer(_p_input_layer),
        p_output_layer(_p_output_layer), layer_param(_layer_param),
        solver_param(_solver_param), p_driver(_p_driver), bias_term(false) {

          // GPU: Use constructor to own data. I.e. allocate this on the device.
          // Update: originally the GPU did own its own data in these cubes, but now
          // that has been refactored to the pbridge, i.e. the pbridge has the cubes
          // which own device data and these just point to those.
          // SHADJIS TODO: For GPU, if we decide to copy 1 image at a time in the
          // batch to the GPU, iB / oB below would change to 1. 
          input_d_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(NULL, iR, iC, iD, iB, p_driver);
          input_g_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(NULL, iR, iC, iD, iB, p_driver);
          output_d_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(NULL, oR, oC, oD, oB, p_driver);
          output_g_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(NULL, oR, oC, oD, oB, p_driver);
        }

    // Second constructor, which does NOT take in a cnn::LayerParameter as a third argument.
    // (Used only for Softmax)
    AbstractBridge<InputLayerDataType, InputLayerLayout, OutputLayerDataType,
      OutputLayerLayout, GPUDriver>(InputLayerType * const _p_input_layer,
          OutputLayerType * const _p_output_layer, GPUDriver * const _p_driver) :
        curr_B(_p_input_layer->p_data_cube->B),
        iR(_p_input_layer->p_data_cube->R), iC(_p_input_layer->p_data_cube->C),
        iD(_p_input_layer->p_data_cube->D), iB(_p_input_layer->p_data_cube->B),
        oR(_p_output_layer->p_data_cube->R), oC(_p_output_layer->p_data_cube->C),
        oD(_p_output_layer->p_data_cube->D), oB(_p_output_layer->p_data_cube->B),
        p_input_layer(_p_input_layer), p_output_layer(_p_output_layer),
        layer_param(NULL), solver_param(NULL), p_driver(_p_driver),
        bias_term(false) {

          // GPU: should not do softmax for now, can change later but then make sure we set iB to 1 below
          assert(false);
          // input_d_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(iR, iC, iD, iB, p_driver);
          // input_g_cube = new LogicalCube<InputLayerDataType, InputLayerLayout>(iR, iC, iD, iB, p_driver);
          // output_d_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(oR, oC, oD, oB, p_driver);
          // output_g_cube = new LogicalCube<OutputLayerDataType, OutputLayerLayout>(oR, oC, oD, oB, p_driver);
        }

    // This needs to be virtual, so we can delete the subclass bridges
    virtual ~AbstractBridge() {
      delete input_d_cube;  input_d_cube  = NULL;
      delete input_g_cube;  input_g_cube  = NULL;
      delete output_d_cube; output_d_cube = NULL;
      delete output_g_cube; output_g_cube = NULL;
    }
    
    // Update the pointer of the input layer's data
    // This function is used to run a different dataset on the network
    // (e.g. for validation set)
    virtual void update_p_input_layer_data_CPU_ONLY(InputLayerDataType * new_data)  {
      p_input_layer->p_data_cube->set_p_data(new_data);
    }
    // Update the pointer of the output layer's gradient
    // This function is used to update the gradients of this layer which
    // come from another source, e.g. a separate network
    virtual void update_p_output_layer_gradient_CPU_ONLY(InputLayerDataType * new_data)  {
      p_output_layer->p_gradient_cube->set_p_data(new_data);
    }
    
    // Get sizes
    virtual size_t get_input_data_size()  {
      return input_d_cube->n_elements;
    }
    virtual size_t get_output_data_size()  {
      return output_d_cube->n_elements;
    }
};
#endif // _INCLUDE_GPUDRIVER

#endif
