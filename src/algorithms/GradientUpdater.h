
#ifndef _GradientUpdater_H
#define _GradientUpdater_H

#include <algorithm>
#include "../LogicalCube.h"
#include "../util.h"
/**
 * The job of a GradientUpdater is simple, it takes
 * as input the gradient, and update the model. It
 * might maintain states by itself to implement
 * algorithm like AdaGrad. However, the caller of
 * this function should not deal with it.
 *
 * Also, this class does not know the structure of
 * the model -- in the eye of a GradientUpdater, they
 * are just a list of parameters.
 *
 **/
template<class DataType, typename DriverClass>
class GradientUpdater {
  public:

    int current_iter; /*< start from 0, every update inc by 1 */
    const int n_elements; /*< # elements in the model. */
    DataType * const p_model; /*< pointer to the model to update */
    const cnn::SolverParameter * const p_solver; /*< object that contains solver parameters. */
    float base_learning_rate, base_regularization;

    DriverClass * const p_driver;
    /**
     * Given a gradient, update the model.
     **/
    virtual void update(DataType * const p_gradient) = 0;

    // SHADJIS TODO: This will not work on the GPU, should use device version
    void reset_zero() {
      // std::fill(p_model, p_model+n_elements, DataType(0.0));
      assert(false);
    }

    float get_stepsize() {
      return base_learning_rate*Util::get_learning_rate(p_solver->lr_policy(), p_solver->base_lr(), p_solver->gamma(),
          current_iter, p_solver->stepsize(), p_solver->power(), p_solver->max_iter());
    }

    void set_current_iter(int _current_iter) {
      current_iter = _current_iter;
    }

    float get_weight_decay() { return p_solver->weight_decay() * base_regularization; }
    
    typedef LogicalCube<DataType, Layout_CRDB> LogicalCubeType;
    virtual LogicalCubeType * get_p_history_updates_cube() { assert(false); }
    virtual void force_device_to_host_history_copy() { assert(false); }
    virtual void force_host_to_device_history_copy() { assert(false); }

    GradientUpdater(int _n_elements, DataType * _p_model, const cnn::SolverParameter * const _p_solver, float _blr, float _br, DriverClass * const _p_driver) :
      current_iter(0), n_elements(_n_elements), p_model(_p_model), p_solver(_p_solver), base_learning_rate(_blr), base_regularization(_br), p_driver(_p_driver) {}

    virtual ~GradientUpdater() {}
};

template<class DataType, class DriverClass>
class SGDGradientUpdater : public GradientUpdater<DataType, DriverClass> {
  public:
   
    using GradientUpdater<DataType, DriverClass>::current_iter;
    using GradientUpdater<DataType, DriverClass>::n_elements;
    using GradientUpdater<DataType, DriverClass>::p_model;  // May be on host or device
    using GradientUpdater<DataType, DriverClass>::p_solver;
    using GradientUpdater<DataType, DriverClass>::get_stepsize;
    using GradientUpdater<DataType, DriverClass>::get_weight_decay;
    using GradientUpdater<DataType, DriverClass>::p_driver;

    typedef LogicalCube<DataType, Layout_CRDB> LogicalCubeType;

    const float momentum;

    // SHADJIS TODO: For SGD update I'm introducing cubes so that
    // device functions are easier to call. Should do this for other
    // updater algorithms too.
    LogicalCubeType *p_history_updates_cube;
    LogicalCubeType *p_history_updates_cube_HOST_COPY;
    
    LogicalCubeType * get_p_history_updates_cube() {
      // If the history is only on the device, return the host copy
      if (p_history_updates_cube_HOST_COPY) {
        assert(!(std::is_same<DriverClass, CPUDriver>::value));
        return p_history_updates_cube_HOST_COPY;
      }
      // The history is on the host
      else {
        assert((std::is_same<DriverClass, CPUDriver>::value));
        return p_history_updates_cube;
      }
    }
    
    void force_device_to_host_history_copy() {
      // If the history is only on the device, we need a copy to the host
      if (p_history_updates_cube_HOST_COPY) {
        assert(!(std::is_same<DriverClass, CPUDriver>::value));
        CPUDriver *tmp_local_cpu_driver = new CPUDriver();
        p_driver->memcpy(p_history_updates_cube_HOST_COPY->get_device_pointer(tmp_local_cpu_driver), p_history_updates_cube->get_device_pointer(p_driver));
        delete tmp_local_cpu_driver;
      }
      // The history is on the host
      else {
        assert((std::is_same<DriverClass, CPUDriver>::value));
      }
    }
    
    void force_host_to_device_history_copy() {
      // If the history is only on the device, we need a copy to the device
      if (p_history_updates_cube_HOST_COPY) {
        assert(!(std::is_same<DriverClass, CPUDriver>::value));
        CPUDriver *tmp_local_cpu_driver = new CPUDriver();
        p_driver->memcpy(p_history_updates_cube->get_device_pointer(p_driver), p_history_updates_cube_HOST_COPY->get_device_pointer(tmp_local_cpu_driver));
        delete tmp_local_cpu_driver;
      }
      // The history is on the host
      else {
        assert((std::is_same<DriverClass, CPUDriver>::value));
      }
    }
    
    LogicalCubeType *p_model_cube;

    SGDGradientUpdater(int _n_elements, DataType * _p_model, const cnn::SolverParameter * const _p_solver, 
        float _blr, float _br, DriverClass * const _p_driver) :
        GradientUpdater<DataType, DriverClass>(_n_elements, _p_model, _p_solver, _blr, _br, _p_driver), momentum(p_solver->momentum()),
        p_history_updates_cube(NULL), p_history_updates_cube_HOST_COPY(NULL), p_model_cube(NULL) {
        
      // Allocate on device
      p_history_updates_cube = new LogicalCubeType(n_elements, 1, 1, 1, p_driver);
      p_driver->sconstant_initialize(p_history_updates_cube->get_device_pointer(p_driver), DataType(0.));
      
      // Also, for saving checkpoints, we need a host copy of this
      if (!std::is_same<DriverClass, CPUDriver>::value) {
        p_history_updates_cube_HOST_COPY = new LogicalCubeType(n_elements, 1, 1, 1);
      }
      
      p_model_cube = new LogicalCubeType(p_model, n_elements, 1, 1, 1, p_driver); 
    }

    ~SGDGradientUpdater() {
      if (p_history_updates_cube_HOST_COPY) {
        delete p_history_updates_cube_HOST_COPY;
      }
      delete p_history_updates_cube;
      delete p_model_cube;
    }

    void update(DataType * const p_gradient) {
      ++current_iter;
      
      PROFILE_ONLY(Timer t; float seconds;)

      const float stepsize = get_stepsize();
      const float lambda   = get_weight_decay();

      if (lambda != 0) {
          if (p_solver->regularization_type() == "L2") {
            p_driver->math_saxpy(n_elements, lambda, p_model, p_gradient);
          }else if (p_solver->regularization_type() == "L1") {
            p_driver->L1_update(n_elements, p_gradient, lambda, p_model);
          }
      }

      PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "        Update step 1:  " << seconds << "\n"; t.restart(); )
      
      // std::cout << "STEPSIZE = " << stepsize << " MOMENTUM = " << momentum << " BASE_LR = "
      //	<< base_learning_rate << " BASE_REG = " << base_regularization << std::endl ;

      //#warning "[BROKEN ABSTRACTION] Using Util when we should be using a driver "

      // This is the right code?
      
      // All these can be combined into a single kernel call
      
      this->p_driver->math_saxpby(n_elements,stepsize, p_gradient, momentum, p_history_updates_cube->get_p_data());
      
      PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "        Update step 2:  " << seconds << "\n"; t.restart(); )
      
      this->p_driver->math_saxpy(n_elements, -1.0, p_history_updates_cube->get_p_data(), p_model_cube->get_p_data());
      
      PROFILE_ONLY(p_driver->device_sync(); seconds = t.elapsed(); std::cout << "        Update step 3:  " << seconds << "\n"; t.restart(); )
      
      /* for (int i=0;i<n_elements;i++) { */
      /* 	p_history_updates[i] = stepsize * p_gradient[i] + momentum * p_history_updates[i]; */
      /* 	p_model[i] -= p_history_updates[i]; */
      /* } */
    }

    float get_momentum() const {
      return momentum;
    }
};

template<class DataType, class DriverClass>
class AdaGradUpdater : public GradientUpdater<DataType, DriverClass> {
  public:

    using GradientUpdater<DataType, DriverClass>::current_iter;
    using GradientUpdater<DataType, DriverClass>::n_elements;
    using GradientUpdater<DataType, DriverClass>::p_model;
    using GradientUpdater<DataType, DriverClass>::p_solver;
    using GradientUpdater<DataType, DriverClass>::get_stepsize;
    using GradientUpdater<DataType, DriverClass>::get_weight_decay;

    DataType * const p_history_updates; /*< Update History */

    AdaGradUpdater(int _n_elements, DataType * _p_model, const cnn::SolverParameter * const _p_solver) :
      GradientUpdater<DataType, DriverClass>(_n_elements, _p_model, _p_solver),
      // SHADJIS TODO: This will not work on the GPU
      p_history_updates(new DataType[_n_elements]) {
        // SHADJIS TODO: This will not work on the GPU
        std::fill(p_history_updates, p_history_updates+n_elements, DataType(0.0));
      }

    ~AdaGradUpdater() {
      delete p_history_updates;
    }

    void update(DataType * const p_gradient) {
      current_iter ++;
      const float stepsize = get_stepsize();
      //const float momentum = p_solver->momentum();
      const float lambda = get_weight_decay();
      const float delta = p_solver->delta();

      if (lambda != 0) {
        // SHADJIS TODO: This will not work on the GPU
        Util::regularize(p_solver->regularization_type(), n_elements, lambda,
            p_gradient, p_model);
      }

      for (int i=0;i<n_elements;i++) {
        // SHADJIS TODO: This will not work on the GPU
        p_history_updates[i] += p_gradient[i]*p_gradient[i];
        p_model[i] -= stepsize / (sqrt(p_history_updates[i])+delta) * p_gradient[i];
        if (i == 0) {
          std::cout << "[0] " << (stepsize / (sqrt(p_history_updates[i])+delta)) << std::endl;
        }
      }
    }
};

template<class DataType, typename DriverClass>
class NesterovUpdater : public GradientUpdater<DataType, DriverClass> {
  public:
    using GradientUpdater<DataType, DriverClass>::current_iter;
    using GradientUpdater<DataType, DriverClass>::n_elements;
    using GradientUpdater<DataType, DriverClass>::p_model;
    using GradientUpdater<DataType, DriverClass>::p_solver;
    using GradientUpdater<DataType, DriverClass>::get_stepsize;
    using GradientUpdater<DataType, DriverClass>::get_weight_decay;

    const float momentum;
    DataType * const p_history_updates; /*< Update History */

    NesterovUpdater(int _n_elements, DataType * _p_model, const cnn::SolverParameter * const _p_solver, DriverClass * const _p_driver) :
        GradientUpdater<DataType, DriverClass>(_n_elements, _p_model, _p_solver, _p_driver), momentum(p_solver->momentum()),
        // SHADJIS TODO: This will not work on the GPU
        p_history_updates(new DataType[_n_elements]) {
        
      // SHADJIS TODO: This will not work on the GPU
      std::fill(p_history_updates, p_history_updates+n_elements, DataType(0.0));
    }

    ~NesterovUpdater() {
      delete p_history_updates;
    }

    void update(DataType * const p_gradient) {
      current_iter++;
      const float stepsize = get_stepsize();
      const float lambda = get_weight_decay();
      //const float delta = p_solver->delta();

      // SHADJIS TODO: This will not work on the GPU
      if (lambda != 0) {
        Util::regularize(p_solver->regularization_type(), n_elements, lambda,
            p_gradient, p_model);
      }

      DataType tmp;
      // SHADJIS TODO: This will not work on the GPU
      for (int i=0;i<n_elements;i++) {
        tmp = p_history_updates[i];
        p_history_updates[i] = stepsize * p_gradient[i] + momentum * tmp;
        p_model[i] -= (1+momentum)*p_history_updates[i] + momentum*tmp;
      }
    }

    float get_momentum() const {
      return momentum;
    }
};

#endif
