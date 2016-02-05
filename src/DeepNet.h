#ifndef _moka_DeepNet_h // SHADJIS TODO: what is moka? Why is it everywhere in the code? Remove all these eventually
#define _moka_DeepNet_h

#include <iostream>
#include <algorithm>
#include <boost/program_options.hpp>
#include <ctime>
#include "LogicalCube.h"
#include "Layer.h"
#include "Connector.h"
#include "Kernel.h"
#include "bridges/AbstractBridge.h"
#include "bridges/MaxPoolingBridge.h"
#include "bridges/ReLUBridge.h"
#include "bridges/ConvolutionBridge.h"
#include "bridges/FullyConnectedBridge.h"
#include "bridges/SoftmaxLossBridge.h"
#include "bridges/LRNBridge.h"
#include "bridges/ParallelizedBridge.h"
#include "bridges/DropoutBridge.h"
#include "bridges/FunnelBridge.h"
#include "bridges/SplitBridge.h"
#include "parser/corpus.h"
#include "DeepNetConfig.h"
#include "util.h"


typedef LogicalCube<DataType_SFFloat, Layout_CRDB> LogicalCubeFloat;
typedef AbstractBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver> Bridge;
typedef SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver> SoftmaxBridge;
typedef std::vector<Bridge *> BridgeVector;

class DeepNet {
  public:
    // static int find_accuracy(const LogicalCubeFloat * const labels, const LogicalCubeFloat * output);

    // static Corpus * load_network(const char * file, const string & data_binary, cnn::SolverParameter & solver_param,
    //     cnn::NetParameter & net_param, BridgeVector & bridges, bool train);

    // static Corpus * read_corpus_from_lmdb(const cnn::NetParameter & net_param, const std::string data_binary, bool train);

    // static void construct_network(BridgeVector & bridges, Corpus & corpus, const cnn::NetParameter & net_param,
    //     const cnn::SolverParameter & solver_param);

    // static float load_and_test_network(const char * file, const std::string data_binary, const std::string model_file);

    // static void load_and_train_network(const char * file, const std::string data_binary, const std::string model_file);
    // computes the output dimension for any convolution layer

    static inline size_t compute_conv_next_layer_dimension(const size_t R_i, const size_t K,
        const size_t padding, const size_t stride ) {
      return (R_i + 2 * padding - K) / stride + 1;      // SHADJIS TODO: This is wrong for pool, since it uses ceil
    }

    static void clean_up(BridgeVector & bridges, Corpus * const corpus) {
      delete bridges.front()->p_input_layer;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        delete (*bridge);
        delete (*bridge)->p_output_layer;
      }
      delete corpus;
    }

    // load training data into Corpus object, return Corpus object
    // Note: we assume that the very first layer in the .protoxt
    // file specifies the data layer
    static Corpus * read_corpus_from_lmdb(const cnn::NetParameter & net_param, bool train) {
      if (train) {
        const cnn::LayerParameter layer_param = net_param.layer(0); // SHADJIS TODO: Should we be hard-coding layer 0 = train?
        string layer_type = layer_param.type();
        std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::toupper);
        if (layer_type == "DATA") {
          if (layer_param.include(0).phase() == 0) { // training phase
            return new Corpus(layer_param);
          }
        }
      } else {
        const cnn::LayerParameter layer_param = net_param.layer(1);
        string layer_type = layer_param.type();
        std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::toupper);
        if (layer_type == "DATA") {
          if (layer_param.include(0).phase() == 1) { // testing phase
            return new Corpus(layer_param);
          }
        }
      }
      std::cout << "No data layer present in prototxt file!" << std::endl;
      assert(false);
      return NULL;
    }
    
    // Run forward pass on network (propagates data from input layer (i.e. input cubes) of bridge 0
    // to output layer (cubes) of final bridge
    static void run_forward_pass(const BridgeVector bridges) {
    
        // Iterate over all bridges, but this might not be 1 at a time
        // if we run some in parallel, so use a while loop
        size_t bridge_idx = 0;
        while (bridge_idx < bridges.size()) {
        
            // Check how many bridges we will run together in this parallel group
            const int num_bridges_in_this_group = bridges[bridge_idx]->get_model_parallelism_group_size();
            assert(num_bridges_in_this_group + bridge_idx <= bridges.size());
            
            // If just 1, then run it as normal
            if (num_bridges_in_this_group == 1) {
                bridges[bridge_idx]->forward();
#ifdef _LAYER_PROFILING
                bridges[bridge_idx]->report_forward();
#endif
            }
            // If more than 1, run them in parallel using C++ threads
            else {
                vector<thread> threads;
                // Create 1 thread per bridge in the group
                for (int b = 0; b < num_bridges_in_this_group; ++b) {
                    // Assert the bridge for this thread it is part of a group of this size
                    assert(bridges[bridge_idx + b]->get_model_parallelism_group_size() == num_bridges_in_this_group);
                    // Create a thread for this bridge
                    threads.push_back(thread([&bridges, bridge_idx, b]() {
                        bridges[bridge_idx + b]->forward();
#ifdef _LAYER_PROFILING
                        bridges[bridge_idx + b]->report_forward();
#endif
                    }));
                }
                // Join
                for (size_t ti = 0; ti < threads.size(); ti++) {
                  threads[ti].join();
                }
            }

            // Move to next bridge
            bridge_idx += num_bridges_in_this_group;
        }
        assert(bridge_idx == bridges.size());
    }

    // Run backward pass on network
    static void run_backward_pass(const BridgeVector bridges) {
    
        // Iterate over all bridges, but this might not be 1 at a time
        // if we run some in parallel, so use a while loop
        int bridge_idx = bridges.size() - 1;
        while (bridge_idx >= 0) {
        
            // Check how many bridges we will run together in this parallel group
            const int num_bridges_in_this_group = bridges[bridge_idx]->get_model_parallelism_group_size();
            assert(bridge_idx - num_bridges_in_this_group >= -1);
            
            // If just 1, then run it as normal
            if (num_bridges_in_this_group == 1) {
                bridges[bridge_idx]->backward();
#ifdef _LAYER_PROFILING
                bridges[bridge_idx]->report_backward();
#endif
            }
            // If more than 1, run them in parallel using C++ threads
            else {
                vector<thread> threads;
                // Create 1 thread per bridge in the group
                for (int b = 0; b < num_bridges_in_this_group; ++b) {
                    // Assert the bridge for this thread it is part of a group of this size
                    assert(bridges[bridge_idx - b]->get_model_parallelism_group_size() == num_bridges_in_this_group);
                    // Create a thread for this bridge
                    threads.push_back(thread([&bridges, bridge_idx, b]() {
                        bridges[bridge_idx - b]->backward();
#ifdef _LAYER_PROFILING
                        bridges[bridge_idx - b]->report_backward();
#endif
                    }));
                }
                // Join
                for (size_t ti = 0; ti < threads.size(); ti++) {
                  threads[ti].join();
                }
            }

            // Move to next bridge
            bridge_idx -= num_bridges_in_this_group;
        }
        assert(bridge_idx == -1);
    }

    // Write in the models of all bridges to a file
    static void write_model_to_file(const BridgeVector bridges, const string model_file) {
      FILE * pFile = fopen (model_file.c_str(), "wb");
      if (!pFile)
        throw std::runtime_error("Error opening " + model_file);

      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        model = (*bridge)->get_model_cube();
        if (model) {
          // If the model is not currently on the host (e.g. it could be on some remote device),
          // force a copy back to the host
          (*bridge)->force_device_to_host_model_copy();
          fwrite(model->get_p_data(), sizeof(DataType_SFFloat), model->n_elements, pFile);
        }
        bias = (*bridge)->get_bias_cube();
        if (bias) {
          (*bridge)->force_device_to_host_bias_copy();
          fwrite(bias->get_p_data(), sizeof(DataType_SFFloat), bias->n_elements, pFile);
        }
      }
      fclose(pFile);
    }

    // Read in the models of all bridges from a file
    static void read_model_from_file(BridgeVector & bridges, const string model_file) {
      FILE * pFile;
      pFile = fopen (model_file.c_str(), "rb");
      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        model = (*bridge)->get_model_cube();
        if (model) {
          size_t num_elements_read = fread(model->get_p_data(), sizeof(DataType_SFFloat), model->n_elements, pFile);
          assert(num_elements_read == model->n_elements);
          (*bridge)->force_host_to_device_model_copy();
        }
        bias = (*bridge)->get_bias_cube();
        if (bias) {
          size_t num_elements_read = fread(bias->get_p_data(), sizeof(DataType_SFFloat), bias->n_elements, pFile);
          assert(num_elements_read == bias->n_elements);
          (*bridge)->force_host_to_device_bias_copy();
        }
      }
      fclose(pFile);
    }

    // Get the total number of parameters (bias + model) in all bridges of the network
    static size_t get_parameter_size(const BridgeVector bridges) {

      size_t total_size = 0;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        model = (*bridge)->get_model_cube();
        if (model) {
          total_size += model->n_elements;
        }
        bias = (*bridge)->get_bias_cube();
        if (bias) {
          total_size += bias->n_elements;
        }
      }
      return total_size;
    }

    // Given a buffer (already allocated on the host), fill it with all the gradients
    // To know how big to make this buffer see get_parameter_size()
    static void get_all_gradients(const BridgeVector bridges, DataType_SFFloat * buffer) {  

      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      size_t total_size = 0;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        model = (*bridge)->get_model_cube();
        if (model) {
          memcpy(buffer + total_size, (*bridge)->get_model_gradient_host(), sizeof(DataType_SFFloat) * model->n_elements);
          total_size += model->n_elements;
        }
        bias = (*bridge)->get_bias_cube();
        if (bias) {
          memcpy(buffer + total_size, (*bridge)->get_bias_gradient_host(),  sizeof(DataType_SFFloat) * bias->n_elements);
          total_size += bias->n_elements;
        }
      }
    }

    // Given a buffer (already allocated on the host), fill it with all the gradients
    // To know how big to make this buffer see get_parameter_size()
    static int get_ith_gradient(const BridgeVector bridges, DataType_SFFloat * buffer, int i) {  

      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      size_t total_size = 0;
      model = (bridges[i])->get_model_cube();
      if (model) {
        memcpy(buffer + total_size, (bridges[i])->get_model_gradient_host(), sizeof(DataType_SFFloat) * model->n_elements);
        total_size += model->n_elements;
      }
      bias = (bridges[i])->get_bias_cube();
      if (bias) {
        memcpy(buffer + total_size, (bridges[i])->get_bias_gradient_host(),  sizeof(DataType_SFFloat) * bias->n_elements);
        total_size += bias->n_elements;
      }
      return total_size;
    }
    static void get_ith_gradient_model_only(const BridgeVector bridges, DataType_SFFloat * buffer, int i) {  
      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      model = (bridges[i])->get_model_cube();
      if (model) {
        memcpy(buffer, (bridges[i])->get_model_gradient_host(), sizeof(DataType_SFFloat) * model->n_elements);
      }
    }
    static void get_ith_gradient_bias_only(const BridgeVector bridges, DataType_SFFloat * buffer, int i) {  
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      bias = (bridges[i])->get_bias_cube();
      if (bias) {
        memcpy(buffer, (bridges[i])->get_bias_gradient_host(),  sizeof(DataType_SFFloat) * bias->n_elements);
      }
    }

    // Given a buffer of all the gradients in the network, update all the models of all the bridges
    static void update_all_models_with_gradients(const BridgeVector bridges, DataType_SFFloat * gradients_concatenated) {

      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      size_t total_size = 0;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
     
        // SHADJIS TODO: I am calling these functions "_CPU", e.g.
        // update_model_with_gradient_CPU. This is because if the bridge
        // has the gradient updates normally on the GPU, then we need to
        // pass in a device pointer. Eventually I should abstract this using
        // a device memory pointer but for now I will assert that the
        // gradient updates for this bridge are on the CPU, and therefore that
        // gradients_concatenated is just a host pointer.
     
        model = (*bridge)->get_model_cube();
        if (model) {
          (*bridge)->update_model_with_gradient_CPU(gradients_concatenated + total_size);
          total_size += model->n_elements;
        }

        bias = (*bridge)->get_bias_cube();
        if (bias) {
          (*bridge)->update_bias_with_gradient_CPU(gradients_concatenated + total_size);
          total_size += bias->n_elements;
        }
      }
            
    }


    // Given a buffer of all the gradients in the network, update all the models of all the bridges
    static void update_ith_models_with_gradients(const BridgeVector bridges, DataType_SFFloat * gradients_concatenated, int i) {

      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      size_t total_size = 0;
     
        // SHADJIS TODO: I am calling these functions "_CPU", e.g.
        // update_model_with_gradient_CPU. This is because if the bridge
        // has the gradient updates normally on the GPU, then we need to
        // pass in a device pointer. Eventually I should abstract this using
        // a device memory pointer but for now I will assert that the
        // gradient updates for this bridge are on the CPU, and therefore that
        // gradients_concatenated is just a host pointer.
     
        model = (bridges[i])->get_model_cube();
        if (model) {
          (bridges[i])->update_model_with_gradient_CPU(gradients_concatenated + total_size);
          total_size += model->n_elements;
        }

        bias = (bridges[i])->get_bias_cube();
        if (bias) {
          (bridges[i])->update_bias_with_gradient_CPU(gradients_concatenated + total_size);
          total_size += bias->n_elements;
        }
            
    }



    // Given a buffer (already allocated on the host), fill it with all the model weights
    // To know how big to make this buffer see get_parameter_size()
    static void get_all_models(const BridgeVector bridges, DataType_SFFloat * buffer) {  

      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      size_t total_size = 0;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        model = (*bridge)->get_model_cube();
        if (model) {
          // If the model is not currently on the host (e.g. it could be on some remote device),
          // force a copy back to the host
          // SHADJIS TODO: If it is on the device we can do a cuda memcpy 
          // This would save doing a copy. However, that is only relevant when the
          // model updates are on the device, which is single-GPU only (1 partition and
          // it is a GPU partition). We can check if that is the case here and then just
          // do a direct memcpy, rather than do two.
          (*bridge)->force_device_to_host_model_copy();
          memcpy(buffer + total_size, model->get_p_data(), sizeof(DataType_SFFloat) * model->n_elements);
          total_size += model->n_elements;
        }
        bias = (*bridge)->get_bias_cube();
        if (bias) {
          (*bridge)->force_device_to_host_bias_copy();
          memcpy(buffer + total_size, bias->get_p_data(), sizeof(DataType_SFFloat) * bias->n_elements);
          total_size += bias->n_elements;
        }
      }
    }

    static int get_ith_models(const BridgeVector bridges, DataType_SFFloat * buffer, int i) {  

      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;

      size_t total_size = 0;
      model = bridges[i]->get_model_cube();
      if(model){
        bridges[i]->force_device_to_host_model_copy();
        memcpy(buffer + total_size, model->get_p_data(), sizeof(DataType_SFFloat) * model->n_elements);
        total_size += model->n_elements;
      }
      bias = bridges[i]->get_bias_cube();
      if (bias) {
        bridges[i]->force_device_to_host_bias_copy();
        memcpy(buffer + total_size, bias->get_p_data(), sizeof(DataType_SFFloat) * bias->n_elements);
        total_size += bias->n_elements;
      }

      return total_size;

    }

    // Like read_model_from_file() but read model from a memory buffer
    static void set_ith_models(const BridgeVector bridges, DataType_SFFloat * models_concatenated, int i) {  

      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;

      size_t total_size = 0;
      model = bridges[i]->get_model_cube();
      if(model){
        memcpy(model->get_p_data(), models_concatenated + total_size, sizeof(DataType_SFFloat) * model->n_elements);
        total_size += model->n_elements;
        bridges[i]->force_host_to_device_model_copy();
      }
      bias = bridges[i]->get_bias_cube();
      if (bias) {
        memcpy(bias->get_p_data(), models_concatenated + total_size, sizeof(DataType_SFFloat) * bias->n_elements);
        total_size += bias->n_elements;
        bridges[i]->force_host_to_device_bias_copy();
      }
    }
    static void set_ith_model_only(const BridgeVector bridges, DataType_SFFloat * models_concatenated, int i) {  
      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      model = bridges[i]->get_model_cube();
      if(model){
        memcpy(model->get_p_data(), models_concatenated, sizeof(DataType_SFFloat) * model->n_elements);
        bridges[i]->force_host_to_device_model_copy();
      }
    }
    static void set_ith_bias_only(const BridgeVector bridges, DataType_SFFloat * models_concatenated, int i) {  
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      bias = bridges[i]->get_bias_cube();
      if (bias) {
        memcpy(bias->get_p_data(), models_concatenated, sizeof(DataType_SFFloat) * bias->n_elements);
        bridges[i]->force_host_to_device_bias_copy();
      }
    }

    // Like read_model_from_file() but read model from a memory buffer
    static void set_all_models(const BridgeVector bridges, DataType_SFFloat * models_concatenated) {  

      LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
      LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
      size_t total_size = 0;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        model = (*bridge)->get_model_cube();
        if (model) {
          // SHADJIS TODO: Could do this with a single cuda memcpy if we need to copy to device later,
          // but now we will do a memcpy here followed by a memcpy if needed to the device (could save 1st one)
          memcpy(model->get_p_data(), models_concatenated + total_size, sizeof(DataType_SFFloat) * model->n_elements);
          total_size += model->n_elements;
          (*bridge)->force_host_to_device_model_copy();
        }
        bias = (*bridge)->get_bias_cube();
        if (bias) {
          memcpy(bias->get_p_data(), models_concatenated + total_size, sizeof(DataType_SFFloat) * bias->n_elements);
          total_size += bias->n_elements;
          (*bridge)->force_host_to_device_bias_copy();
        }
      }
    }

    static int find_accuracy(const LogicalCubeFloat * const labels, const LogicalCubeFloat * output, bool verbose = false) {
      const float * actual_data = output->get_p_data();
      const float * expected_label = labels->get_p_data();
      int top_k = 1;
      float accuracy = 0;
      int num = output->B;
      int dim = output->D;
      vector<float> maxval(top_k+1);
      vector<int> max_id(top_k+1);
      for (int i = 0; i < num; ++i) {
        // Top-k accuracy
        std::vector<std::pair<float, int> > data_vector;
        for (int j = 0; j < dim; ++j) {
          data_vector.push_back(
              std::make_pair(actual_data[i * dim + j], j));
        }
        std::partial_sort(
            data_vector.begin(), data_vector.begin() + top_k,
            data_vector.end(), std::greater<std::pair<float, int> >());
        // check if true label is in top k predictions
        for (int k = 0; k < top_k; k++) {
          //if (verbose) {
          //  std::cout << "Guess = " << data_vector[k].second << " | Actual = " << static_cast<int>(expected_label[i]) << "\n";
          //}
          if (data_vector[k].second == static_cast<int>(expected_label[i])) {
            ++accuracy;
            break;
          }
        }
      }
      return accuracy;
      //cout << "Accuracy: " << (accuracy / num) << std::endl;
    }

    // This takes in the bridge vector (which has been initialized to be empty in load_and_train_network)
    // and builds up a list of bridges in the vector in the order in which they will be executed in the forward
    // pass. Only the bridges variable is modified.
    static void construct_network(BridgeVector & bridges, Corpus & corpus, const cnn::NetParameter & net_param,
        const cnn::SolverParameter & solver_param) {
      CPUDriver * const driver = new CPUDriver(); // SHADJIS TODO: delete this later or put on stack
      const int hw_concurrency = std::thread::hardware_concurrency();
      assert(hw_concurrency > 0);
      
      size_t input_R = corpus.n_rows, input_C = corpus.n_cols, input_D = corpus.dim, B = corpus.mini_batch_size;

      // Create the Logical Cubes for the initial data layer
      LogicalCubeFloat * prev_data = new LogicalCubeFloat(corpus.images->physical_get_RCDslice(0), input_R, input_C, input_D, B);
      LogicalCubeFloat * prev_grad = new LogicalCubeFloat(input_R, input_C, input_D, B);

      std::vector<Layer<DataType_SFFloat, Layout_CRDB> *> prev_layers, next_layers;
      prev_layers.push_back(new Layer<DataType_SFFloat, Layout_CRDB>(prev_data, prev_grad));

      const size_t num_layers = net_param.layer_size();

      Bridge * bridge = NULL;
      LogicalCubeFloat * next_data = NULL;
      LogicalCubeFloat * next_grad = NULL;
      Layer<DataType_SFFloat, Layout_CRDB> * next_layer = NULL;

      size_t output_R = input_R, output_C = input_C, output_D = input_D;
      // SHADJIS TODO: Currently this just skips the backward data calculation
      // for the first conv but it could be applied to all layers before it too,
      // or the first fc and all layers before it (if the net has no conv).
      // No backward pass wrt data is needed for the first fc or conv layer
      // (whichever is first) as well as everything before it.
      bool before_first_weight_layer = true;
      
      // -----------------------------------------------------------------------
      // Scheduler
      //
      // Eventually there will be a global scheduler which schedules data 
      // movement across bridges. For now, part of this scheduler will be 
      // inside the following loop. The portion of the scheduler that schedules
      // within bridges is in ParallelizedBridge.
      // 
      // When 2 PBridges are consecutive, e.g. a conv bridge is followed be a
      // relu bridge, and if both bridges are on the GPU, then there is no need
      // to copy back and forth from the host between bridges. This can be done
      // currently in PBridge but the bridge needs to know:
      //   1. Device information + data from the previous PBridge, and
      //   2. Whether to share data with the next PBridge
      // Since each PBridge is isolated from the others, those signals are 
      // passed to the bridges through the loop below

      // First, make (initially empty) vectors storing device information of
      // the previous bridge. These will be updated each iteration by creating
      // a bridge and then reading the result from that bridge.
      //
      // SHADJIS TODO: But these are only defined for PBridges, not other types
      // of bridges. All bridges in the loop below are just AbstractBridges, i.e.
      // these vectors may not exist for that bridge. This is motivation to just
      // merge AbstractBridge and ParallelizedBridge, but can do that later.
      size_t prev_num_partitions_CPU = 0;
      std::vector<size_t> prev_GPU_batch_sizes;
      std::vector<int> prev_gpu_to_device_id_map;
      // Edit: Making this a vector of vectors. The outer vector is for each group,
      // the inner vector is the # (CPU+GPU) partitions for the bridge of that group.
      std::vector< std::vector< LogicalCube<DataType_SFFloat, Layout_CRDB> *> > prev_data_cubes_higher_per_group;
      std::vector< std::vector< LogicalCube<DataType_SFFloat, Layout_CRDB> *> > prev_grad_cubes_higher_per_group;
      // Also by default resize these to 1
      // This is because there is always 1 "group" (the default network). The 1 element of the vector (i.e.
      // the 1 interior cube vector) will be uninitialized, but it will never be used anyway until this 
      // vector gets properly filled because prev_data_cubes_higher_per_group is only used for GPUs (1 or many) 
      // to share input layer of current bridge with output layer of prev bridge (which does not exist for first bridge),
      // and only does when share_pointer_with_prev_bridge is true, which it never will be for the first layer since 
      // the previous layer (which does not exist) has no GPU bridges.
      //
      // More notes:
      //  - Recall the role of prev_data_cubes_higher_per_group: It is to allow GPU
      //    bridges (1 GPU or many) to share input data with the previous layer. This is normally
      //    just a single vector of cubes but if we have multiple groups, we need to keep that
      //    vector per group, so it ends up being a vector of vectors.
      //  - prev_data_cubes_higher_per_group is just the output layer of the previous bridge.
      //    Higher means top or output, and lower means bottom or input. It has nothing to do
      //    with conv lowering / lifting (i.e. the name prev_data_cubes_higher_per_group is bad)
      //  - The grouping is always like this:   1 1 1 1 N N N N FC, i.e. zero or more 1's followed 
      //    by zero or more of a single group size N (cannot change), and ending at FC (FC always
      //    inserts a funnel). Normal networks are just 1 1 ... 1 1 FC, with no N
      //  - n_previous_groups = 1 in first iteration of this loop (start with 1 data layer),
      //    but prev_data_cubes_higher_per_group has size 0 (it is a new network, so there is
      //    no out data cube from the previous layer (which does not exist). So we resize to 1. 
      //  - These cubes only are used for GPUs (1 or many) to share data, but they do not decide
      //    whether or not to share -- that is done by the GPU properties of the prev bridge
      //  - Right now, grouping of 2 for example will make 2 bridges, but then execute them serially.
      //    E.g. if you have 2 GPUs, and 2 groups, 2 pbridges will be created -- the first uses 2 gpus 
      //    and once it finishes the second uses 2 gpus. It might be better to use 1 GPU each since 
      //    this avoids model copies back to host, and do both in parallel. In fact, that is how model 
      //    parallelism is implemented: using groups and parallel threads to launch each bridge in the group
      //    (see comment in the model parallelism section of the code)
      //
      prev_data_cubes_higher_per_group.resize(1);
      prev_grad_cubes_higher_per_group.resize(1);
      
      // Second, also make a vector of the bridges from the last iteration
      BridgeVector pbridges_from_last_iteration;
      BridgeVector pbridges_from_this_iteration;
      
      // SHADJIS TODO: Currently we only use the vectors above to adjust 
      // PBridges. This does not affect those bridges which are not part of a
      // PBridge (softmax, funnel). Need to also make those
      // pbridges, so that they too can avoid copies / extra allocations. Then,
      // everything is a PBridge, so either merge PBridge and AbstractBridge or
      // move certain scheduler elements to the AbstractBridge.

      // -----------------------------------------------------------------------

      for (size_t i_layer = 0; i_layer < num_layers; ++i_layer) {

        const cnn::LayerParameter layer_param = net_param.layer(i_layer);
        string layer_type = layer_param.type();
        std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::toupper);

        if (layer_type == "ACCURACY")
        {
            // Ignore, we do this automatically
            continue;
        }

        // SHADJIS TODO: Some layers, e.g. ReLU and dropout, have the same top
        // and bottom layers and therefore don't need to allocate any output cubes.
        // For now, we will avoid this allocation only in PBridge (since it matters
        // more for the GPU), i.e. just for ReLU/dropout. When we support others
        // (e.g. funnel may also have same top/bottom) as PBridges, then those will
        // also benefit from not needing extra cube allocations (and also not needing
        // copies to/from host if entire network is on GPU)
        bool share_input_output_layer = false;
        if (layer_param.top_size() && layer_param.bottom_size() &&
            layer_param.top_size() == layer_param.bottom_size() &&
            layer_param.top(0) == layer_param.bottom(0))
        {
            share_input_output_layer = true;
        }
        const size_t n_previous_groups = prev_layers.size();

        if (layer_type != "DATA") {
            
          if (layer_type == "CONVOLUTION")
          {
              const size_t K = layer_param.convolution_param().kernel_size(),
                    padding = layer_param.convolution_param().pad(),
                    stride = layer_param.convolution_param().stride();
              size_t grouping = layer_param.convolution_param().group();

              /*
               * This is for syntax compatibility with Caffe about grouping.
               * In the protocol buf file, if layer A, B, C has grouping 1, 2, 2,
               * in Caffe, A is also partition'ed into two groups... Without
               * the following fix, we need the syntax to be 2, 2, 2 to do
               * the same thing. The following fix makes sure the syntax is
               * consistent.
               */
              size_t next_conv_layer = i_layer;
              while ((++next_conv_layer) < num_layers) {
                const cnn::LayerParameter next_layer_param = net_param.layer(next_conv_layer);
                string next_layer_type = next_layer_param.type();
                std::transform(next_layer_type.begin(), next_layer_type.end(), next_layer_type.begin(), ::toupper);
                if (next_layer_type == "CONVOLUTION") {
                  size_t next_grouping = next_layer_param.convolution_param().group();
                  if (grouping == 1 && next_grouping != 1) {
                    grouping = next_grouping;
                  }
                  break;
                }
              }

              std::cout << "Constructing CONV bridge with Grouping = " << grouping <<
                " (# Input Grouping=" << n_previous_groups << ")" << std::endl;

              output_R = compute_conv_next_layer_dimension(input_R, K, padding, stride),
                       output_C = compute_conv_next_layer_dimension(input_C, K, padding, stride),
                       output_D = layer_param.convolution_param().num_output();
              if (output_D % grouping != 0) {
                std::cout << "ERROR: Currently we only support the output depth \% grouping == 0." << std::endl;
                assert(false);
              }
              output_D /= grouping;

              if (grouping == n_previous_groups) {
              
                // Assert that our vectors which keep track of GPU pointers are the correct size
                assert(prev_data_cubes_higher_per_group.size() == n_previous_groups);
                assert(prev_grad_cubes_higher_per_group.size() == n_previous_groups);
                // if input group == output group, then for each
                // input group, create a separate bridge and a
                // seperate output bridge
                for (size_t i = 0; i < n_previous_groups; i++) {
                  // for each group, create bridges
                  next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                  next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                  next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

                  // SHADJIS TODO: No need to pass partitions and threads as an argument, abstract it within
                  // the pbridge. Or for the user to be able specify # partitions, then also do not pass 
                  // this as an argument, instead let it be part of the same config file as GPU
                  bridge = new ParallelizedBridge<DataType_SFFloat, ConvolutionBridge>
                           (prev_layers[i], next_layer, &layer_param, &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1, // TODO: need a CMD line option here -- but currently we do not have the interface to do that.
                           prev_num_partitions_CPU, prev_GPU_batch_sizes, prev_gpu_to_device_id_map, prev_data_cubes_higher_per_group[i], prev_grad_cubes_higher_per_group[i],
                           share_input_output_layer);
                  bridge->name = layer_param.name();
                  if (before_first_weight_layer) {
                      bridge->needs_to_calc_backward_grad = false;
                  }
                  bridges.push_back(bridge);
                  next_layers.push_back(next_layer);
                  pbridges_from_this_iteration.push_back(bridge);
                }
              } else {

                assert(prev_data_cubes_higher_per_group.size() == 1);
                assert(prev_grad_cubes_higher_per_group.size() == 1);
              
                // This is where the split happens (n_previous_groups starts at 1 if this is conv1)
                if (grouping != 1 && n_previous_groups == 1) {
                  // In this case, we fork the single input group into multile output groups
                  //
                  // Note that below we will pass prev_data_cubes_higher_per_group[0] into the new pbridge, 
                  // since all new bridges in the upcoming group have identical input data (only smaller depths)
                  //
                  // However -- they all still need to put their data gradients to different locations so we
                  // can sum them all up at the end (i.e. in bw pass, each bridge fills its input_g_cubes, but
                  // all groups now need to sum their data gradients so we can pass back a single data gradient)
                  // For this, we insert a SplitBridge. The FW pass does nothing since we are going to just
                  // re-use the pointers. But the BW pass needs to sum from each group.
                  // 
                  // Optimization: We do not need a split bridge if we are before the first weight layer.
                  // I.e. if we never use the backward data gradient of this bridge, no need for a Split.
                  if (!before_first_weight_layer) {
                  
                    // Add a split bridge
                    
                    // SHADJIS TODO: Ideally, at this split we could still share pointers even if the bridge 
                    // before the split was on the GPU (or many GPUs). I.e. before the split we might have all 
                    // of our data on GPUs 1-4, and the upcoming group will use the same batch parallelism per 
                    // bridge in the group, so we can share input data pointers (i.e. each bridge in the group 
                    // can share its input data cube with the output data cube of the prev bridge's output layer).
                    // However, we cannot share the input gradient cube of the group's bridges with the output
                    // gradient cube of the previous bridge since we need to sum up all of the gradients going
                    // backwards in the split. For this reason I will just make this a CPU bridge, although we
                    // could save a copy in the FW pass.
                    // Note: this is not true of a split before model parallelism since then we need all the
                    // data to go to each GPU, which is never would have previously been if we did data 
                    // parallelism (and if we did model parallelism before, then we must merge anyway in order
                    // to have the full data and calculate the correct gradient).
                  
                    std::cout << "  First Constructing SPLIT bridge with input grouping 1 (# Output Grouping=" << grouping << ")" << std::endl;
                    // Note: Similarly to how funnel never uses the input layer, split never uses the output layer
                    // Therefore we can just make empty cubes
                    next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(NULL, input_R, input_C, input_D, B);
                    next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(NULL, input_R, input_C, input_D, B);
                    next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
                    // On the other hand, the input layer needs to be prev_layers[0] so we can copy that directly to each output cube in the fw pass
                    assert(prev_layers.size() == 1);
                    Layer<DataType_SFFloat, Layout_CRDB> * input_layer_of_split_bridge = prev_layers[0];
                    prev_layers.clear();
                    bridge = new SplitBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>(input_layer_of_split_bridge,
                        next_layer, &layer_param, &solver_param, driver);
                    // Now we have our bridge, but we need to create the actual output layers, one per bridge in the upcoming group.
                    // Recall each bridge has an input and output layer. Each layer consists of a data and gradient cube.
                    // - The input layer of the split bridge is the same as the output data cube of the previous bridge's output layer,
                    //   prev_layers[0] (now called input_layer_of_split_bridge). In the FW pass we read data from that layer's data cube and in
                    //   the bw pass we fill gradients into that layer's grad cube.
                    // - The output layer of the split bridge is unused, and instead we have many output layers (p_output_layers),
                    //   one per bridge in the upcoming group. 
                    // In the fw pass, we just need to pass the data cube from the split bridge's input layer to each output data cube.
                    // I.e. in the FW pass we just make sure this input data cube is the same (pointer) as all the output cubes.
                    // In the backwards pass, we need to read the gradients from each of the output layers' gradient cubes, sum them,
                    // and write that to the input gradient cube. This means we need to allicate space for gradient but not data cubes
                    // each output layer of the split bridge. Do that here:
                    for (size_t i = 0; i < grouping; ++i) {
                      // Do not allocate
                      next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_layer_of_split_bridge->p_data_cube->get_p_data(), input_R, input_C, input_D, B);
                      // Allocate
                      next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
                      next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
                      ((SplitBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>*)bridge)->p_output_layers.push_back(next_layer);
                      prev_layers.push_back(next_layer);
                    }
                    bridge->name = "SPLIT";
                    bridges.push_back(bridge);
                    
                    // Since we are inserting a cpu bridge, also clear all information about past pbridge
                    pbridges_from_last_iteration.clear();
                    pbridges_from_this_iteration.clear();
                    prev_GPU_batch_sizes.clear();
                    prev_gpu_to_device_id_map.clear();
                    prev_data_cubes_higher_per_group.clear();
                    prev_grad_cubes_higher_per_group.clear();
                    // Resize these to 1 again (see comment above)
                    prev_data_cubes_higher_per_group.resize(1);
                    prev_grad_cubes_higher_per_group.resize(1);
                    
                    // Now create each conv bridge for the upcoming group
                    assert(prev_layers.size() == grouping);
                    for (size_t i = 0; i < grouping; ++i) {
                      // for each group, create bridges
                      next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                      next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                      next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
                  
                      bridge = new ParallelizedBridge<DataType_SFFloat, ConvolutionBridge>
                               (prev_layers[i], next_layer, &layer_param, &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1, // TODO: need a CMD line option here -- but currently we do not have the interface to do that.
                               prev_num_partitions_CPU, prev_GPU_batch_sizes, prev_gpu_to_device_id_map, prev_data_cubes_higher_per_group[0], prev_grad_cubes_higher_per_group[0],
                               share_input_output_layer);
                      bridge->name = layer_param.name();
                      assert(!before_first_weight_layer);
                      bridges.push_back(bridge);
                      next_layers.push_back(next_layer);
                      pbridges_from_this_iteration.push_back(bridge);
                    }
                  }
                  // Otherwise, there is no split bridge. This happens e.g. if the grouping is 2 2 2 2 2
                  // (or equivalently 1 2 2 2 2, since we look at the next one due to code change above).
                  // In this case the grouping happens right away but there is no need to caclulate the
                  // backwards gradient for that first conv so we never need to do a backwards split.
                  // Therefore all the bridges in the group we are about to create can all just share a
                  // single input layer. If this was a split bridge (equivalently, if we had to calculate
                  // backward data grad for this bridge) then we could not share the input layer across
                  // all of these because they would all need to put their backwards grad cubes somewhere
                  // (but for the fw pass they could).
                  else {
                    assert(prev_layers.size() == 1);
                    for (size_t i = 0; i < grouping; i++) {
                      // for each group, create bridges
                      next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                      next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                      next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
                  
                      bridge = new ParallelizedBridge<DataType_SFFloat, ConvolutionBridge>
                               (prev_layers[0], next_layer, &layer_param, &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1, // TODO: need a CMD line option here -- but currently we do not have the interface to do that.
                               prev_num_partitions_CPU, prev_GPU_batch_sizes, prev_gpu_to_device_id_map, prev_data_cubes_higher_per_group[0], prev_grad_cubes_higher_per_group[0],
                               share_input_output_layer);
                      bridge->name = layer_param.name();
                      assert(before_first_weight_layer);
                      if (before_first_weight_layer) {
                          bridge->needs_to_calc_backward_grad = false;
                      }
                      bridges.push_back(bridge);
                      next_layers.push_back(next_layer);
                      pbridges_from_this_iteration.push_back(bridge);
                    }
                  }
                } else {
                  std::cout << "ERROR: Currently we do not support the case where input group is " << n_previous_groups
                    << " and output group is " << grouping << " for CONV layer" << std::endl;
                  assert(false);
                }
              }
              // -------------- Scheduler Update ----------------
              // SHADJIS TODO: Rather than copy this everywhere refactor into a function.
              // Or, instead, just make every bridge into a pbridge so we can refactor the
              // code below to the end of the loop. Then can also re-use bridges instead
              // of creating pbridges_from_this_iteration. Or, can move the function
              // set_share_pointer_with_next_bridge into abstract bridge (but only
              // overload for pbridge) and then calling this for non-pbridges will have
              // no effect, so the code can still be refactored.
              //
              // SHADJIS TODO: Also, if the 2 bridges are sharing device data pointers,
              // then there is no need to copy the data to or from the host, so the cube
              // allocations above (the cubes own their own data) are unnecessary since
              // those cubes are never used anywhere in pbridge.
              assert(pbridges_from_this_iteration.size());
              bool bridge_shares_data_with_prev_bridge = 
                  pbridges_from_this_iteration[0]->get_share_pointer_with_prev_bridge();
              if (bridge_shares_data_with_prev_bridge) {
                for (size_t it = 0; it < pbridges_from_last_iteration.size(); ++it) {
                  pbridges_from_last_iteration[it]->set_share_pointer_with_next_bridge(true);
                }
              }
              pbridges_from_last_iteration = pbridges_from_this_iteration;
              prev_GPU_batch_sizes = pbridges_from_this_iteration[0]->get_GPU_batch_sizes();
              prev_num_partitions_CPU = pbridges_from_this_iteration[0]->get_num_partitions_CPU();
              prev_gpu_to_device_id_map = pbridges_from_this_iteration[0]->get_used_gpu_to_device_id_map();
              prev_data_cubes_higher_per_group.clear();
              prev_grad_cubes_higher_per_group.clear();
              for (size_t i=0; i<pbridges_from_this_iteration.size(); ++i) {
                prev_data_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_data_cubes_higher());
                prev_grad_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_grad_cubes_higher());
              }
              pbridges_from_this_iteration.clear();
              // ----------End of Scheduler Update --------------
              before_first_weight_layer = false;
          }
          else if (layer_type == "INNERPRODUCT")
          {
              // Always insert a funnel for FC
              if (n_previous_groups != 1) {
                // if the previous group of this fully-connected layer contains multiple
                // groups, then it's the time to unify them. To do this, we introduce a
                // bridge whose only role is a funnel
                std::cout << "Constructing FUNNEL bridge with output grouping 1 (# Input Grouping=" << n_previous_groups << ")" << std::endl;
                output_R = input_R; output_C = input_C; output_D = input_D * n_previous_groups;
                next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
                // Note: Funnel never uses the input layer, i.e. we pass in prev_layers[0] but this p_input_layer is never used,
                // since instead we use p_input_layers which contains all of them
                bridge = new FunnelBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>(prev_layers[0],
                    next_layer, &layer_param, &solver_param, driver);
                for (size_t i = 0; i < n_previous_groups; i++) {
                  ((FunnelBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>*)bridge)->p_input_layers.push_back(prev_layers[i]);
                }
                bridge->name = "FUNNEL";
                bridges.push_back(bridge);
                input_D = output_D;
                prev_layers.clear();
                prev_layers.push_back(next_layer);
                
                // Since we are inserting a cpu bridge, also clear all information about past pbridge
                pbridges_from_last_iteration.clear();
                pbridges_from_this_iteration.clear();
                prev_GPU_batch_sizes.clear();
                prev_gpu_to_device_id_map.clear();
                prev_data_cubes_higher_per_group.clear();
                prev_grad_cubes_higher_per_group.clear();
                // Resize these to 1 again (see comment above)
                prev_data_cubes_higher_per_group.resize(1);
                prev_grad_cubes_higher_per_group.resize(1);
              }

              // =============================================================================================================
              // MODEL PARALLELISM
              // =============================================================================================================
              
              // -------------------------------------------------------------------------------------------------------------
              // SHADJIS TODO: Refactor Model Parallelism
              // -------------------------------------------------------------------------------------------------------------
              // Now that we have the funnel, we need to create the FC bridge.
              // 
              // Normally we would create a single FC parallelized bridge. This should handle data parallelism or model
              // parallelism, i.e. whatever is defined in the layer_param object (gpu_0_batch_proportion, gpu_2_depth_proportion,
              // etc.), the pbridge should handle. So all we should need to do for a pbridge is create the bridge using the
              // input layer, output layer, and params as is currently the case (as well as the other things we have now), and
              // it will just fill the output layer with the right data in fw or update gradients in input grad layer and 
              // update model as well during bw. I.e. we don't care in DeepNet.h what the pbridge is doing (model parallelism,
              // data parallelism, gpus, whatever), all we do is call pbridge->forward() and it fills the appropriate layers in.
              // 
              // Curently this isn't the case for model parallelism -- to implement model parallelism quickly I re-used all of
              // the grouping code. Grouping is just partitioning by depth of model, whereas data parallelism is partitioning
              // by batch. Since grouping already exists I just re-use that, but then run the bridges in parallel (normally 
              // the bridges in each group runs serially) using threads inside run_forward_pass and run_backward_pass. 
              // 
              // So for now we are going to parse the layer_param to check for model parallelism (AKA partition by depth) inside
              // DeepNet.h and then create multiple pbridges which we will launch in parallel. Eventually, this should just
              // create a single pbridge, and that bridge can look up what type of parallelism to do and do it, abstracted
              // from the caller. That requires a few changes which I listed here:
              // 
              // Changes to ParallelizedBridge:
              // 
              // - Constructor:
              //   - Read protobuf to check for data or model parallelism, i.e. batch partition or depth partition (if both, can give an error)
              //   - Data parallelism is same as now (same for FW and BW below). Rest of notes are for model parallelism:
              //   - Model parallelism: The loop over batches would be replaced with a loop over depth partitions, and we would 
              //     give all the data in the input layer to each input cube (not just a subset of the batches). However, 
              //     we would make the output cubes have smaller D (e.g. D/4 if 4 GPUs)
              // - FW:
              //   - Same as now, but afterwards implement something like a funnel (simple -- copy data to host and merge) so that the
              //     returned output at the end is on the host and merged (not partitioned by depth) (do copies to host in parallel like we do now)
              // - BW:
              //   - Once each parallel sub-bridge finishes, do model updates in parallel on the device (no copy to host)
              //   - Then copy the input gradient cubes to the host and merge (combine by depth) (do copies to host in parallel like we do now)
              //
              // Summary: Below we parse layer parameters to check for model parallelism, then create multiple pbridges
              // as if it was grouping, and then use threads during fw/bw pass to run bridges in a group in parallel.
              // In the future we should instead create a single parallelized bridge and do all this in there.
              // -------------------------------------------------------------------------------------------------------------

              // The R and C dimensions for a fully connected layer are always 1 x 1
              output_R = output_C = 1;
              output_D = layer_param.inner_product_param().num_output();

              // -------------------------------------------------------------------------------------------------------------
              // Determine number of groups
              // -------------------------------------------------------------------------------------------------------------
              
              size_t number_of_model_parallel_groups = 0;
              
              // First, read layer parameters to check how many groups to make, as well as the depth proportion on each
              // SHADJIS TODO: This is a simplified version from Pbridge.h update_scheduler_partitions_for_curr_B().
              // When we move this code there, we can re-use that.
              // Note: there is no depth_proportion on CPU because on CPU data and model parallelism is the same
              // (it is just a big GEMM using all the threads -- whether we partition data by B or D and then do
              // GEMM should not matter, but can measure this to see).
              // SHADJIS TODO: Eventually we will want to support CPU + GPU, like we do for data parallelism, and
              // therefore support any proportion. This is not hard but we need to:
              //   - Get proportions using code like in PBridge's update_scheduler_partitions_for_curr_B
              //   - Create pbridges with different depths each in the output cube (see code below)
              //   - Modify funnel to merge even if depths are not all the same (easy, see commend in there)
              //
              std::vector <float> GPU_depth_proportions;
              GPU_depth_proportions.push_back(layer_param.gpu_0_depth_proportion());
              GPU_depth_proportions.push_back(layer_param.gpu_1_depth_proportion());
              GPU_depth_proportions.push_back(layer_param.gpu_2_depth_proportion());
              GPU_depth_proportions.push_back(layer_param.gpu_3_depth_proportion());
              float portion_of_depth_per_GPU = 0.;   // Eventually will not be needed, once depth partitions can be nonuniform
              std::vector <int> GPUs_used_for_model_parallelism;  // Redundant given info above but makes code easier to read
              
              // Iterate over these and for now assert the proportions are correct
              // Also keep the sum of all the GPU depth proportions
              float sum_of_gpu_depth_proportions = 0.;
              for (size_t gpu_idx=0; gpu_idx < GPU_depth_proportions.size(); ++gpu_idx) {
                // For now there is no point to model parallelism on 1 GPU
                if (GPU_depth_proportions[gpu_idx] == 1.) {
                  std::cout << "Error: GPU has depth proportion set to 1.0, but model parallelism requires 2 or more GPUs.\n" 
                            << "if running on CPU or single GPU, use batch proportion instead.\n";
                  assert(false);
                }
                // Check this portion divides output depth exactly
                float float_number_on_this_GPU = float(output_D) * GPU_depth_proportions[gpu_idx];
                if (float_number_on_this_GPU != int(float_number_on_this_GPU)) {
                  std::cout << "Error: Currently model parallelism must divide depth exactly. Error for output_D = " 
                            << output_D << ", proportion = " << GPU_depth_proportions[gpu_idx] << "\n";
                  assert(false);
                }
                // Check also that all portions are the same (for now)
                // If portion_of_depth_per_GPU has not been set yet (no GPU so far has any depth allocated to it)
                if (portion_of_depth_per_GPU == 0) {
                    portion_of_depth_per_GPU = GPU_depth_proportions[gpu_idx];
                }
                // Else GPUs are being used, so assert they are all using the same proportion
                else if (GPU_depth_proportions[gpu_idx] > 0.) {
                  if (portion_of_depth_per_GPU != GPU_depth_proportions[gpu_idx]) {
                    std::cout << "Error: Currently each GPU must have matching depth proportions\n";
                    assert(false);
                  }
                }
                
                // If this GPU does have some usage, keep track of its number
                if (GPU_depth_proportions[gpu_idx] > 0.) {
                    number_of_model_parallel_groups += 1;
                    GPUs_used_for_model_parallelism.push_back(gpu_idx);
                }
                
                sum_of_gpu_depth_proportions += GPU_depth_proportions[gpu_idx];
              }
              // Finally, check that either all on CPU or all on GPU (for now no CPU + GPU)
              if (sum_of_gpu_depth_proportions != 0 && sum_of_gpu_depth_proportions != 1) {
                std::cout << "Error: Currently model parallelism must be completely on the GPU or on the CPU. Soon sharing will be supported.\n";
                assert(false);
              }
              // If using GPUs for model parallelism
              if (sum_of_gpu_depth_proportions == 1) {
                assert(portion_of_depth_per_GPU < 1.);   // For now, no model parallelism on a single GPU (makes no sense)
                assert(portion_of_depth_per_GPU > 0.);
                assert(number_of_model_parallel_groups > 1);
                assert(number_of_model_parallel_groups == GPUs_used_for_model_parallelism.size());
              }
              // Otherwise no model parallelism on GPU (could still be using any # of GPUs for data parallelism,
              // but that is for the pbridge to handle)
              else {
                assert(portion_of_depth_per_GPU == 0);
                assert(number_of_model_parallel_groups == 0);
              }
              
              // -------------------------------------------------------------------------------------------------------------
              // Now we know if everything is on the GPU or the CPU, so we can create bridges
              // -------------------------------------------------------------------------------------------------------------
              
              // -------------------------------------------------------------------------------------------------------------
              // Normal case: No model parallelism, so just create a single pbridge. This is the normal case.
              // -------------------------------------------------------------------------------------------------------------
              if (number_of_model_parallel_groups == 0) {
              
                assert(portion_of_depth_per_GPU == 0.);
              
                std::cout << "Constructing FC layer " << "(# Input Grouping=" << 1 << ")" << std::endl;
              
                next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
              
                assert(prev_data_cubes_higher_per_group.size() == 1);
                assert(prev_grad_cubes_higher_per_group.size() == 1);
                
                bridge = new ParallelizedBridge<DataType_SFFloat, FullyConnectedBridge>
                         (prev_layers[0], next_layer, &layer_param, &solver_param, driver, min<size_t>(1, corpus.mini_batch_size), hw_concurrency,
                         prev_num_partitions_CPU, prev_GPU_batch_sizes, prev_gpu_to_device_id_map, prev_data_cubes_higher_per_group[0], prev_grad_cubes_higher_per_group[0],
                         share_input_output_layer);
              
                bridge->name = layer_param.name();
                bridge->run_with_n_threads = hw_concurrency;  // TODO: Add a better abstraction here. // SHADJIS TODO: Is this still needed?
                if (before_first_weight_layer) {
                    // bridge->needs_to_calc_backward_grad = false;
                }
                bridges.push_back(bridge);
                next_layers.push_back(next_layer);
                pbridges_from_this_iteration.push_back(bridge);
                
                // -------------- Scheduler Update ----------------
                assert(pbridges_from_this_iteration.size());
                bool bridge_shares_data_with_prev_bridge = 
                    pbridges_from_this_iteration[0]->get_share_pointer_with_prev_bridge();
                if (bridge_shares_data_with_prev_bridge) {
                  for (size_t it = 0; it < pbridges_from_last_iteration.size(); ++it) {
                    pbridges_from_last_iteration[it]->set_share_pointer_with_next_bridge(true);
                  }
                }
                pbridges_from_last_iteration = pbridges_from_this_iteration;
                prev_GPU_batch_sizes = pbridges_from_this_iteration[0]->get_GPU_batch_sizes();
                prev_num_partitions_CPU = pbridges_from_this_iteration[0]->get_num_partitions_CPU();
                prev_gpu_to_device_id_map = pbridges_from_this_iteration[0]->get_used_gpu_to_device_id_map();
                prev_data_cubes_higher_per_group.clear();
                prev_grad_cubes_higher_per_group.clear();
                for (size_t i=0; i<pbridges_from_this_iteration.size(); ++i) {
                  prev_data_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_data_cubes_higher());
                  prev_grad_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_grad_cubes_higher());
                }
                pbridges_from_this_iteration.clear();
                // ----------End of Scheduler Update --------------
              }
              // -------------------------------------------------------------------------------------------------------------
              // Model Parallelism Case: 
              // -------------------------------------------------------------------------------------------------------------
              // Now we need to create multiple bridges
              // For now, we know this will use GPUs (CPU case is handled above)
              // Also, we assume that the data is on the host, i.e. do not share with previous bridge
              else {
                // -----------------------------------------------------------------------------------------------------------
                // Add a Split Bridge
                // -----------------------------------------------------------------------------------------------------------
                // The split bridge is on the CPU. We will always do a copy to/from CPU before model parallelism because 
                // model parallelism only makes sense on > 1 GPU and we need all the data on that GPU. Since all the data
                // never exists on multiple GPUs at the same time (because we use data parallelism), we need a copy of the
                // data here (even if we had a funnel just now because there was grouping before, we need both that funnel
                // and this split bridge since we need to merge all the data before doing model parallelism)
                // See all the comments in the other split bridge above, where it is used for grouping, for 
                // more information.
                // SHADJIS TODO: In the original split above for conv (for grouping) I added a small optimization to
                // avoid the split when the conv is the first layer with a model, and therefore when there is no
                // backward gradient needed. That same optimization can be done here but I did not do it yet.
                // More generally, needs_to_calc_backward_grad should be used inside every bridge to potentially
                // skip the backwards gradient, but currently is only in conv.
                std::cout << "Constructing SPLIT bridge for upcoming model-parallel FC. Splitting depth from 1 to " 
                          << number_of_model_parallel_groups << " partitions" << std::endl;
                
                // Note: Similarly to how funnel never uses the input layer, split never uses the output layer
                // Therefore we can just make empty cubes
                next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(NULL, input_R, input_C, input_D, B);
                next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(NULL, input_R, input_C, input_D, B);
                next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
                
                // On the other hand, the input layer needs to be prev_layers[0] so we can copy that directly to each output cube in the fw pass
                assert(prev_layers.size() == 1);
                Layer<DataType_SFFloat, Layout_CRDB> * input_layer_of_split_bridge = prev_layers[0];
                prev_layers.clear();
                bridge = new SplitBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>(input_layer_of_split_bridge,
                    next_layer, &layer_param, &solver_param, driver);
                    
                // See comment in previous split (conv) bridge: now we need to create output layers, one per bridge in the upcoming 
                // model-parallel group.
                for (size_t i = 0; i < number_of_model_parallel_groups; ++i) {
                  // Do not allocate
                  next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_layer_of_split_bridge->p_data_cube->get_p_data(), input_R, input_C, input_D, B);
                  // Allocate
                  next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
                  next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
                  ((SplitBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>*)bridge)->p_output_layers.push_back(next_layer);
                  prev_layers.push_back(next_layer);
                }
                bridge->name = "SPLIT";
                bridges.push_back(bridge);
                
                // Since we are inserting a cpu bridge, also clear all information about past pbridge
                pbridges_from_last_iteration.clear();
                pbridges_from_this_iteration.clear();
                prev_GPU_batch_sizes.clear();
                prev_gpu_to_device_id_map.clear();
                prev_data_cubes_higher_per_group.clear();
                prev_grad_cubes_higher_per_group.clear();
                // Resize these to 1 again (see comment above)
                prev_data_cubes_higher_per_group.resize(1);
                prev_grad_cubes_higher_per_group.resize(1);
                
                assert(prev_layers.size() == number_of_model_parallel_groups);

                // -----------------------------------------------------------------------------------------------------------
                // Create each bridge in the model parallel FC bridge
                // -----------------------------------------------------------------------------------------------------------
                
                std::cout << "Constructing Model-Parallel FC layer with Total Depth = " << output_D << std::endl;
                
                for (size_t i = 0; i < number_of_model_parallel_groups; i++) {
                
                  // Read the depth for this bridge
                  int gpu_idx = GPUs_used_for_model_parallelism[i];
                  int output_D_partition = output_D * GPU_depth_proportions[gpu_idx];
                  assert(GPU_depth_proportions[gpu_idx] == portion_of_depth_per_GPU);
                  assert(portion_of_depth_per_GPU > 0);
                  std::cout << "  Constructing partition of model-parallel FC layer with partial depth = " << output_D_partition << std::endl;
                  
                  // ---------------------------------------------------------------------------------------------------------
                  // Update the solver to only use this GPU, if this is a GPU bridge
                  // ---------------------------------------------------------------------------------------------------------
                  // SHADJIS TODO: Fix hack: This is a hack now, I am going to make a new layer_param object and
                  // change the GPU allocations. This is a hack and it will be fixed when we do model
                  // parallelism properly inside the pbridge. Also for now this has a memory leak
                  // (this will also be fixed when this is part of pbridge since no need for separate layer param objects)
                  cnn::LayerParameter * layer_param_tmp = new cnn::LayerParameter(layer_param); // Copy constructor
                  // Set this layer to use data parallelism on a single GPU
                  layer_param_tmp->set_gpu_0_batch_proportion(0.);
                  layer_param_tmp->set_gpu_1_batch_proportion(0.);
                  layer_param_tmp->set_gpu_2_batch_proportion(0.);
                  layer_param_tmp->set_gpu_3_batch_proportion(0.);
                  cnn::InnerProductParameter * inner_product_param_tmp = const_cast<cnn::InnerProductParameter *>(&layer_param_tmp->inner_product_param());
                  inner_product_param_tmp->set_num_output(output_D_partition); // Not sure if this changes original object or just the copy
                  if (gpu_idx == 0) {
                    layer_param_tmp->set_gpu_0_batch_proportion(1.0);
                  } else if (gpu_idx == 1) {
                    layer_param_tmp->set_gpu_1_batch_proportion(1.0);
                  } else if (gpu_idx == 2) {
                    layer_param_tmp->set_gpu_2_batch_proportion(1.0);
                  } else {
                    assert(gpu_idx == 3);
                    layer_param_tmp->set_gpu_3_batch_proportion(1.0);
                  }
                  
                  // ---------------------------------------------------------------------------------------------------------
                  // Now create the bridge for this group
                  // ---------------------------------------------------------------------------------------------------------
                  next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D_partition, B);
                  next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D_partition, B);
                  next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

                  assert(prev_data_cubes_higher_per_group.size() == 1);
                  assert(prev_grad_cubes_higher_per_group.size() == 1);
                
                  // SHADJIS TODO: Can also use a new driver for each, i.e. rather than driver, pass in new CPUDriver().
                  // This didn't cause any problems but since we will run these pbridges in parallel, using the same driver
                  // means that the drivers' internal class variables will be shared. Currently drivers have no variables but they may later.
                  bridge = new ParallelizedBridge<DataType_SFFloat, FullyConnectedBridge>
                           (prev_layers[i], next_layer, layer_param_tmp, &solver_param, driver, min<size_t>(1, corpus.mini_batch_size), hw_concurrency,
                           prev_num_partitions_CPU, prev_GPU_batch_sizes, prev_gpu_to_device_id_map, prev_data_cubes_higher_per_group[0], prev_grad_cubes_higher_per_group[0],
                           share_input_output_layer);

                  bridge->name = layer_param_tmp->name();
                  bridge->run_with_n_threads = hw_concurrency;  // TODO: Add a better abstraction here.
                  if (before_first_weight_layer) {
                      // bridge->needs_to_calc_backward_grad = false;
                  }
                  bridges.push_back(bridge);
                  next_layers.push_back(next_layer);
                  pbridges_from_this_iteration.push_back(bridge);
                  
                  // Update this bridge to set its model parallelism group size
                  bridge->set_model_parallelism_group_size(number_of_model_parallel_groups);
                }
                
                // -----------------------------------------------------------------------------------------------------------
                // We finished constructing our fc bridges
                // Now make a funnel again to restore group size to 1
                // -----------------------------------------------------------------------------------------------------------
                std::cout << "Constructing FUNNEL bridge to merge " << number_of_model_parallel_groups << " depth partitions of model-parallel FC bridge" << std::endl;
                next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
                // Now that now this funnel gets next_layers as input since it needs to merge the layers we just made above
                // Note: Funnel never uses the input layer, i.e. we pass in prev_layers[0] but this p_input_layer is never used,
                // since instead we use p_input_layers which contains all of them
                bridge = new FunnelBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>(next_layers[0],
                    next_layer, &layer_param, &solver_param, driver);
                for (size_t i = 0; i < number_of_model_parallel_groups; i++) {
                  ((FunnelBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>*)bridge)->p_input_layers.push_back(next_layers[i]);
                }
                bridge->name = "FUNNEL";
                bridges.push_back(bridge);
                prev_layers.clear();  // This gets done at the end of this loop anyway
                next_layers.clear();
                next_layers.push_back(next_layer);
                
                // Since we are inserting a cpu bridge, also clear all information about past pbridge
                // (These were done above and are redundant here but make it clearer in the code what is happening)
                pbridges_from_last_iteration.clear();
                pbridges_from_this_iteration.clear();
                prev_GPU_batch_sizes.clear();
                prev_gpu_to_device_id_map.clear();
                prev_data_cubes_higher_per_group.clear();
                prev_grad_cubes_higher_per_group.clear();
                // Resize these to 1 again (see comment above)
                prev_data_cubes_higher_per_group.resize(1);
                prev_grad_cubes_higher_per_group.resize(1);
              }
              
              // =============================================================================================================
              // End of Model Parallelism
              // -------------------------------------------------------------------------------------------------------------
              // - The rest of the implementation happens in run_forward/backward_pass
              // =============================================================================================================
              
              before_first_weight_layer = false;
          }
          else if (layer_type == "POOLING")
          {
              std::cout << "Constructing MAXPOOLING " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

              const size_t K = layer_param.pooling_param().kernel_size(), stride = layer_param.pooling_param().stride();

              output_R = compute_conv_next_layer_dimension(input_R, K, 0, stride),
                       output_C = compute_conv_next_layer_dimension(input_C, K, 0, stride);

              assert(prev_data_cubes_higher_per_group.size() == n_previous_groups);
              assert(prev_grad_cubes_higher_per_group.size() == n_previous_groups);
              
              for (size_t i = 0; i < n_previous_groups; i++) {
                // input_D same as output_D
                next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, input_D, B);
                next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, input_D, B);
                next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

                bridge = new ParallelizedBridge<DataType_SFFloat, MaxPoolingBridge>(prev_layers[i], next_layer, &layer_param,
                           &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1,
                           prev_num_partitions_CPU, prev_GPU_batch_sizes, prev_gpu_to_device_id_map, prev_data_cubes_higher_per_group[i], prev_grad_cubes_higher_per_group[i],
                           share_input_output_layer);
                bridge->name = layer_param.name();
                bridges.push_back(bridge);
                next_layers.push_back(next_layer);
                pbridges_from_this_iteration.push_back(bridge);
              }
              // -------------- Scheduler Update ----------------
              assert(pbridges_from_this_iteration.size());
              bool bridge_shares_data_with_prev_bridge = 
                  pbridges_from_this_iteration[0]->get_share_pointer_with_prev_bridge();
              if (bridge_shares_data_with_prev_bridge) {
                for (size_t it = 0; it < pbridges_from_last_iteration.size(); ++it) {
                  pbridges_from_last_iteration[it]->set_share_pointer_with_next_bridge(true);
                }
              }
              pbridges_from_last_iteration = pbridges_from_this_iteration;
              prev_GPU_batch_sizes = pbridges_from_this_iteration[0]->get_GPU_batch_sizes();
              prev_num_partitions_CPU = pbridges_from_this_iteration[0]->get_num_partitions_CPU();
              prev_gpu_to_device_id_map = pbridges_from_this_iteration[0]->get_used_gpu_to_device_id_map();
              prev_data_cubes_higher_per_group.clear();
              prev_grad_cubes_higher_per_group.clear();
              for (size_t i=0; i<pbridges_from_this_iteration.size(); ++i) {
                prev_data_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_data_cubes_higher());
                prev_grad_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_grad_cubes_higher());
              }
              pbridges_from_this_iteration.clear();
              // ----------End of Scheduler Update --------------
          }
          else if (layer_type == "RELU")
          {
              // input_[R,C,D] is the same as output_[R,C,D]

              std::cout << "Constructing RELU layer " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

              assert(prev_data_cubes_higher_per_group.size() == n_previous_groups);
              assert(prev_grad_cubes_higher_per_group.size() == n_previous_groups);
              
              for (size_t i=0;i<n_previous_groups;i++) {

                next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
                next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
                next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

                bridge = new ParallelizedBridge<DataType_SFFloat, ReLUBridge>(prev_layers[i], next_layer, &layer_param,
                           &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1,
                           prev_num_partitions_CPU, prev_GPU_batch_sizes, prev_gpu_to_device_id_map, prev_data_cubes_higher_per_group[i], prev_grad_cubes_higher_per_group[i],
                           share_input_output_layer);
                bridge->name = layer_param.name();

                bridges.push_back(bridge);
                next_layers.push_back(next_layer);
                pbridges_from_this_iteration.push_back(bridge);
              }
              // -------------- Scheduler Update ----------------
              assert(pbridges_from_this_iteration.size());
              bool bridge_shares_data_with_prev_bridge = 
                  pbridges_from_this_iteration[0]->get_share_pointer_with_prev_bridge();
              if (bridge_shares_data_with_prev_bridge) {
                for (size_t it = 0; it < pbridges_from_last_iteration.size(); ++it) {
                  pbridges_from_last_iteration[it]->set_share_pointer_with_next_bridge(true);
                }
              }
              pbridges_from_last_iteration = pbridges_from_this_iteration;
              prev_GPU_batch_sizes = pbridges_from_this_iteration[0]->get_GPU_batch_sizes();
              prev_num_partitions_CPU = pbridges_from_this_iteration[0]->get_num_partitions_CPU();
              prev_gpu_to_device_id_map = pbridges_from_this_iteration[0]->get_used_gpu_to_device_id_map();
              prev_data_cubes_higher_per_group.clear();
              prev_grad_cubes_higher_per_group.clear();
              for (size_t i=0; i<pbridges_from_this_iteration.size(); ++i) {
                prev_data_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_data_cubes_higher());
                prev_grad_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_grad_cubes_higher());
              }
              pbridges_from_this_iteration.clear();
              // ----------End of Scheduler Update --------------
          }
          else if (layer_type == "LRN")
          {
              // input_[R,C,D] is the same as output_[R,C,D]

              std::cout << "Constructing LRN layer " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

              assert(prev_data_cubes_higher_per_group.size() == n_previous_groups);
              assert(prev_grad_cubes_higher_per_group.size() == n_previous_groups);
              
              for (size_t i=0;i<n_previous_groups;i++) {

                next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
                next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
                next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

                bridge = new ParallelizedBridge<DataType_SFFloat, LRNBridge>(prev_layers[i], next_layer, &layer_param,
                           &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1,
                           prev_num_partitions_CPU, prev_GPU_batch_sizes, prev_gpu_to_device_id_map, prev_data_cubes_higher_per_group[i], prev_grad_cubes_higher_per_group[i],
                           share_input_output_layer);
                bridge->name = layer_param.name();

                bridges.push_back(bridge);
                next_layers.push_back(next_layer);
                
                pbridges_from_this_iteration.push_back(bridge);
              }
              // -------------- Scheduler Update ----------------
              assert(pbridges_from_this_iteration.size());
              bool bridge_shares_data_with_prev_bridge = 
                  pbridges_from_this_iteration[0]->get_share_pointer_with_prev_bridge();
              if (bridge_shares_data_with_prev_bridge) {
                for (size_t it = 0; it < pbridges_from_last_iteration.size(); ++it) {
                  pbridges_from_last_iteration[it]->set_share_pointer_with_next_bridge(true);
                }
              }
              pbridges_from_last_iteration = pbridges_from_this_iteration;
              prev_GPU_batch_sizes = pbridges_from_this_iteration[0]->get_GPU_batch_sizes();
              prev_num_partitions_CPU = pbridges_from_this_iteration[0]->get_num_partitions_CPU();
              prev_gpu_to_device_id_map = pbridges_from_this_iteration[0]->get_used_gpu_to_device_id_map();
              prev_data_cubes_higher_per_group.clear();
              prev_grad_cubes_higher_per_group.clear();
              for (size_t i=0; i<pbridges_from_this_iteration.size(); ++i) {
                prev_data_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_data_cubes_higher());
                prev_grad_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_grad_cubes_higher());
              }
              pbridges_from_this_iteration.clear();
              // ----------End of Scheduler Update --------------
          }
          else if (layer_type == "DROPOUT")
          {
              std::cout << "Constructing DROPOUT layer " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

              assert(prev_data_cubes_higher_per_group.size() == n_previous_groups);
              assert(prev_grad_cubes_higher_per_group.size() == n_previous_groups);
              
              for (size_t i=0;i<n_previous_groups;i++) {

                next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
                next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
                next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

                // SHADJIS TODO: I made the max threads 4 below because it was faster than 1 or 16.
                // For these smaller bridges it is usually slower to use all the threads, but need to measure.
                // Then can do something similar for ReLU, etc.
                bridge = new ParallelizedBridge<DataType_SFFloat, DropoutBridge>(prev_layers[i], next_layer, &layer_param,
                           &solver_param, driver, min<size_t>(4, corpus.mini_batch_size), 1,
                           prev_num_partitions_CPU, prev_GPU_batch_sizes, prev_gpu_to_device_id_map, prev_data_cubes_higher_per_group[i], prev_grad_cubes_higher_per_group[i],
                           share_input_output_layer);
                bridge->name = layer_param.name();

                bridges.push_back(bridge);
                next_layers.push_back(next_layer);
                pbridges_from_this_iteration.push_back(bridge);
              }
              // -------------- Scheduler Update ----------------
              assert(pbridges_from_this_iteration.size());
              bool bridge_shares_data_with_prev_bridge = 
                  pbridges_from_this_iteration[0]->get_share_pointer_with_prev_bridge();
              if (bridge_shares_data_with_prev_bridge) {
                for (size_t it = 0; it < pbridges_from_last_iteration.size(); ++it) {
                  pbridges_from_last_iteration[it]->set_share_pointer_with_next_bridge(true);
                }
              }
              pbridges_from_last_iteration = pbridges_from_this_iteration;
              prev_GPU_batch_sizes = pbridges_from_this_iteration[0]->get_GPU_batch_sizes();
              prev_num_partitions_CPU = pbridges_from_this_iteration[0]->get_num_partitions_CPU();
              prev_gpu_to_device_id_map = pbridges_from_this_iteration[0]->get_used_gpu_to_device_id_map();
              prev_data_cubes_higher_per_group.clear();
              prev_grad_cubes_higher_per_group.clear();
              for (size_t i=0; i<pbridges_from_this_iteration.size(); ++i) {
                prev_data_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_data_cubes_higher());
                prev_grad_cubes_higher_per_group.push_back(pbridges_from_this_iteration[i]->get_grad_cubes_higher());
              }
              pbridges_from_this_iteration.clear();
              // ----------End of Scheduler Update --------------
          }
          else if (layer_type == "SOFTMAXWITHLOSS")
          {
              std::cout << "Constructing SOFTMAX layer " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

              // input_[R,C,D] is the same as output_[R,C,D]
              if (n_previous_groups != 1) {
                std::cout << "ERROR: Currently, we only support FC layer to connect " <<
                  "between multiple input groups to a single output group." << std::endl;
                assert(false);
              }

              next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
              next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
              next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
              // must be initialized to point to next mini batch
              LogicalCubeFloat * const labels = new LogicalCubeFloat(NULL, 1, 1, 1, B);

              bridge = new SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat,
                     Layout_CRDB, CPUDriver>(prev_layers[0], next_layer, labels, driver);
              bridge->name = layer_param.name();

              bridges.push_back(bridge);
              next_layers.push_back(next_layer);
          }
          else
          {
              std::cout << "This layer type is not supported: "<< layer_type << "!" << std::endl;
              assert(false);
          }

          input_R = output_R, input_C = output_C, input_D = output_D;

          /**
           * Swap next_layers with prev_layers and empty next;
           */
          prev_layers.clear();
          for (size_t i = 0; i < next_layers.size(); i++) {
            prev_layers.push_back(next_layers[i]);
          }
          next_layers.clear();
        }
      }
    }

    // Here, we train our CNN: we iterate over the vector of bridges, forwards and backward for each batch size.
    // Right now, we do this in a single-thread fashion. TODO: Create a Scheduler class, that schedules workers
    // for each batch size, so that we can perform these forward and backward passes in parallel.
    static void train_network(const BridgeVector & bridges, Corpus & corpus, const cnn::NetParameter & net_param,
        const cnn::SolverParameter & solver_param, const string input_model_file, const string snapshot_file_name,
        Corpus & val_corpus, bool time_iterations = false) {

      SoftmaxBridge * const softmax = (SoftmaxBridge *) bridges.back();
      Bridge * const first = (Bridge *) bridges.front();

      softmax->p_data_labels->set_p_data(corpus.labels->physical_get_RCDslice(0));
      LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;

      float t_load;
      float t_forward;
      float t_backward;
      float t_pass;

      Timer t_total;

#ifdef _LAYER_PROFILING
      const int display_iter = 1;
#else
      const int display_iter = solver_param.display();
#endif
      const int snapshot = solver_param.snapshot();

      // SHADJIS TODO: Support solver_param.test_interval(), i.e. every few training
      // iterations do testing (validation set). For now we can keep the batch size
      // the same during testing but this also does not need to be the case.
      const int test_interval = solver_param.test_interval();

      // Read the number of iterations to run. This is the number of times we will
      // update weights, i.e. the number of mini-batches we will run.
      const size_t num_batch_iterations = solver_param.max_iter();
      
      // It is necessary to open the reader before loading data to initialize
      // cursor, transaction and environment data
      corpus.OpenLmdbReader();

      // Keep track of the image number in the dataset we are on
      size_t current_image_location_in_dataset = 0;
      size_t current_epoch = 0;    
      // std::cout << "EPOCH: " << current_epoch << std::endl;
      float loss = 0.;
      float accuracy = 0.;
    
      // Run for max_iter iterations
      for (size_t batch = 0; batch < num_batch_iterations; ++batch) {

        Timer t;
        Timer t2;
        
        // SHADJIS TODO: corpus.last_batch_size is unused, can remove now
        // SHADJIS TODO: This should be done in parallel with the network execution if slow (measure)
        // SHADJIS TODO: curr_B is unused now in every bridge, can remove it or plan to support variable batch size

        // Read in the next mini-batch from db
        size_t rs = corpus.LoadLmdbData();

        // Note that the implementation of labels has changed.  Since we are reading the lmbd for every
        // iteration, we get the label in the data so now the labels and images objects are parallel
        // TODO? : Perhaps this is a good reason to merge them into a single object 
        // Since labels are in sync they will also only be of size mini_batch_size
        assert(softmax->p_data_labels->get_p_data() == corpus.labels->get_p_data());

        // If we read less than we expected, read the rest from the beginning 
        size_t num_images_left_to_read = corpus.mini_batch_size - rs;
        if (num_images_left_to_read > 0) {
            // Increment epoch
            ++current_epoch;
            // Simply reset the cursor so the next load will start from the start of the lmdb
            corpus.ResetCursor();
            
            // Passing in rs allows us to say that we already filled rs spots in images
            // and now we want to start from that position and complete the set up to mini_batch_size
            // Eg. Minibatch is 10.  We read 2 images and hit the end of the mldb.  After reseting the
            // cursor above we can just tell the load function to start from index 2 and continue
            size_t rs2 = corpus.LoadLmdbData(rs);
            assert(rs2 == num_images_left_to_read);
            
            // The corpus.labels object was also updated above so we need to check that
            // the pointer is still consistent
            assert(softmax->p_data_labels->get_p_data() == corpus.labels->get_p_data());
        }
        
        current_image_location_in_dataset += corpus.mini_batch_size;
        if (current_image_location_in_dataset >= corpus.n_images) {
          current_image_location_in_dataset -= corpus.n_images;
        }

        t_load = t.elapsed();

        t.restart();
        // initialize input_data for this mini batch
        // Ce: Notice the change here compared with the master branch -- this needs to be refactored
        // to make the switching between this and the master branch (that load everything in memory)
        // dynamically and improve code reuse.
        float * const mini_batch = corpus.images->physical_get_RCDslice(0);
        assert(input_data->get_p_data() == mini_batch);

        softmax->reset_loss();

        // forward pass
        run_forward_pass(bridges);

        t_forward = t.elapsed();

        loss += (softmax->get_loss() / float(corpus.mini_batch_size));
        accuracy += float(DeepNet::find_accuracy(softmax->p_data_labels, (*--bridges.end())->p_output_layer->p_data_cube)) / float(corpus.mini_batch_size);

        // backward pass
        t.restart();
        run_backward_pass(bridges);
        t_backward = t.elapsed();

        t_pass = t2.elapsed();

        // Check if we should print batch status
        // Edit: Instead we will make display_iter print the average since
        // the previous display, since this seems more useful
        if ( (batch+1) % display_iter == 0 ) {
          float learning_rate = Util::get_learning_rate(solver_param.lr_policy(), solver_param.base_lr(), solver_param.gamma(),
            batch+1, solver_param.stepsize(), solver_param.power(), solver_param.max_iter());
          
          std::cout << "Training Status Report (Epoch " << current_epoch << " / Mini-batch iter " << batch << "), LR = " << learning_rate << std::endl;
          std::cout << "  \033[1;32m";
          std::cout << "Loss & Accuracy [Average of Past " << display_iter << " Iterations]\t" << loss/float(display_iter) << "\t" << float(accuracy)/(float(display_iter));
          std::cout << "\033[0m" << std::endl;
          loss = 0.;
          accuracy = 0.;
          
          if (time_iterations) {
            std::cout << "\033[1;31m";
            std::cout << "  Iteration Time Status Report (seconds)" << std::endl;
            std::cout << "    Loading Data:  " << t_load << std::endl;
            std::cout << "    Forward Pass:  " << t_forward << std::endl;
            std::cout << "    Backward Pass: " << t_backward << std::endl;
            std::cout << "    Total:         " << t_pass << std::endl;
            std::cout << "\033[0m";
          }
          
        }
        // Check if we should run validation
        if (test_interval > 0 && (batch+1) % test_interval == 0) {
            std::cout << "Validation/Test Status Report (Epoch " << current_epoch << " / Mini-batch iter " << batch << ")" << std::endl;
            // Switch dataset to val
            std::cout << "  \033[1;36m";
            bridges[0]->update_p_input_layer_data_CPU_ONLY(val_corpus.images->physical_get_RCDslice(0));
            DeepNetConfig::train_ = false;
            test_network(bridges, val_corpus, net_param, solver_param, time_iterations);
            // Switch dataset back to train
            // reset the softmax data labels to the corpus labels instead of the test labels
            softmax->p_data_labels->set_p_data(corpus.labels->physical_get_RCDslice(0));
            bridges[0]->update_p_input_layer_data_CPU_ONLY(corpus.images->physical_get_RCDslice(0));
            DeepNetConfig::train_ = true;
            std::cout << "    [Run on entire validation set]\033[0m" << std::endl;
        }
        // Check if we should write a snapshot
        if (snapshot > 0 && (batch+1) % snapshot == 0) {
          time_t rawtime;
          struct tm * timeinfo;
          char buffer[80];
          time (&rawtime);
          timeinfo = localtime(&rawtime);
          strftime(buffer,80,"%d-%m-%Y-%I-%M-%S",timeinfo);
          std::string str(buffer);
          std::string snapshot_name;
          
          if (snapshot_file_name == "NA") {
            snapshot_name = "trained_model.bin." + str;
          } else {
            snapshot_name = snapshot_file_name + "." + str;
          }
          write_model_to_file(bridges, snapshot_name);
          std::cout << "======= Writing snapshot " << snapshot_name << " =======\n";
        }
      }
      
      // This frees any handles we have to the lmdb and free allocated internal objects.
      // Note that corpus.images and corpus.labels are still usable
      corpus.CloseLmdbReader();
      std::cout << "Total Time (seconds): " << t_total.elapsed() << std::endl;
    }

      static Corpus * load_network(const char * file, cnn::SolverParameter & solver_param,
          cnn::NetParameter & net_param, BridgeVector & bridges, bool train) {

        // not necessary if being called from load_and_(train|test)_network,
        // but necessary for certain tests
        DeepNetConfig::train_ = train;

        if (Parser::read_proto_from_text_file(file, &solver_param) &&
            Parser::read_net_params_from_text_file(solver_param.net(), &net_param)) {
          Corpus * corpus = DeepNet::read_corpus_from_lmdb(net_param, train);

#ifdef DEBUG
          std::string corpus_type = train ? "train" : "test";
          std::cout << "Corpus " << corpus_type << " loaded" << std::endl;
          std::cout << "CORPUS NUM IMAGES: " << corpus->n_images << std::endl;
          std::cout << "CORPUS NUM ROWS: " << corpus->n_rows << std::endl;
          std::cout << "CORPUS NUM COLS: " << corpus->n_cols << std::endl;
          std::cout << "CORPUS NUM CHANNELS: " << corpus->dim << std::endl;
          std::cout << "CORPUS MINI BATCH SIZE: " << corpus->mini_batch_size << std::endl;
          assert(corpus->n_images >= corpus->mini_batch_size);
          // std::cout << "CORPUS NUM MINI BATCHES: " << corpus->num_mini_batches << std::endl;
          // std::cout << "CORPUS LAST BATCH SIZE: " << corpus->last_batch_size << std::endl;
#endif

          DeepNet::construct_network(bridges, *corpus, net_param, solver_param);

          return corpus;
        } else {
          throw std::runtime_error("Error parsing the solver.protoxt file or train_val.txt file");
          return NULL;
        }
      }


      static float test_network(const BridgeVector & bridges, Corpus & corpus, const cnn::NetParameter & net_param,
          const cnn::SolverParameter & solver_param, bool time_iterations = false) {

        // TODO: we need a more general AbstractLossBridge
        SoftmaxBridge * const softmax = (SoftmaxBridge *) bridges.back();
        Bridge * const first = (Bridge *) bridges.front();

        // set softmax data to point to our corpus labels buffer
        softmax->p_data_labels->set_p_data(corpus.labels->physical_get_RCDslice(0));
        LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;

        corpus.OpenLmdbReader();
        
        // SHADJIS TODO: Here I could check the size of the corpus (test or validation set),
        // divide by solver_param.test_iter(), and use that as the mini-batch
        // size to run the testing for test_iter. Instead, for now I will just use whatever 
        // batch size is defined in the train/test prototxt file. 
        //
        // The reason for this is that this function (test_network) is called both for validation 
        // and testing. For validation, it is called from within training). During training, the 
        // bridges are constructed for only the mini-batch size: currently for training we wrap 
        // around the dataset, i.e. the last mini-batch may be a different size so we wrap around 
        // and keep every batch size the same always. An equivalent solution is to run the last 
        // mini-batch as a different size, but currently some refactoring is needed to make the 
        // last batch a different size since we have kernels and connectors defined in the bridge 
        // constructors which are of a fixed size (each bridge allocates cubes in the constructor 
        // and does not free them until the destructor).
        // 
        // So because:
        //   1) in training the bridges are constructed for the training mini-batch size,
        //   2) bridges do not support variable sized batches yet,and
        //   3) test_network() could be called from training to run validation (with the same
        //      bridges used in training),
        // currently this function uses the mini-batch size for the testing batch size, i.e. what
        // is in the train/test prototxt, not the solver prototxt. 
        // So eventually the test set mini-batch size will be defined like this:
        //const int test_iter = solver_param.test_iter();
        //const size_t test_mini_batch_size = corpus.n_images / test_iter;
        // But for now to change the test set mini-batch size just change the train/test prototxt.
        
        //float t_load;
        //float t_forward;
        //float t_pass;
        float total_loss = 0.;
        int total_accuracy = 0;
        // const int display_iter = solver_param.test_display();

        // num_mini_batches - 1, because we need one more iteration for the final mini batch
        // (the last mini batch may not be the same size as the rest of the mini batches)
        // Ignore remainder for now, since should try to use a test batch size that divides test set
        const size_t num_batch_iterations = corpus.n_images / corpus.mini_batch_size;
        for (size_t batch = 0, corpus_batch_index = 0; batch < num_batch_iterations; ++batch,
            corpus_batch_index += corpus.mini_batch_size) {

          //Timer t;
          //Timer t2;

          size_t num_elements_read = corpus.LoadLmdbData();
          assert(num_elements_read == corpus.mini_batch_size);
          //t_load = t.elapsed();
          //t.restart();
          float * const mini_batch = corpus.images->physical_get_RCDslice(0);
          assert(input_data->get_p_data() == mini_batch);

          softmax->reset_loss();

          // check that label pointer is correct
          assert(softmax->p_data_labels->get_p_data() == corpus.labels->get_p_data());
          // forward pass
          run_forward_pass(bridges);
          
          //t_forward = t.elapsed();

          float loss = (softmax->get_loss() / corpus.mini_batch_size);
          total_loss += loss;
          int batch_accuracy = DeepNet::find_accuracy(softmax->p_data_labels, softmax->p_output_layer->p_data_cube);
          total_accuracy += batch_accuracy;

          //t_pass = t2.elapsed();

          //if (time_iterations) {
          //  std::cout << "\033[1;31m";
          //  std::cout << "  Iteration Time Status Report (seconds)" << std::endl;
          //  std::cout << "    Loading Data:  " << t_load << std::endl;
          //  std::cout << "    Forward Pass:  " << t_forward << std::endl;
          //  std::cout << "    Total:         " << t_pass << std::endl;
          //  std::cout << "\033[0m";
          //}

        }
        float acc = (1.0*total_accuracy/(num_batch_iterations*corpus.mini_batch_size));
        std::cout << "Loss = " << total_loss / float(num_batch_iterations) << ", Accuracy " << acc;
        corpus.CloseLmdbReader();
        return acc;
      }

      // We expect this to be called from main,
      // it takes in a const char * argument (most likely
      // from arvg[1]) that represents the .prototxt file
      // which specifies the *solver* for the network, not
      // the network configuration file itself.
      //
      // There are three steps involved in training the network:
      //
      // 1) Load the necessary training data into a Corpus object,
      //    which will contain both the data itself, and the correct
      //    labels.
      //
      // 2) Construct the necessary Bridge, Layer, and LogicalCube
      //    objects to represent the network. A network should be
      //    represented as an STL vector of Bridge pointers, so that we
      //    can easily compute the forward pass and the backward pass.
      //
      // 3) For iter = 0 -> max_iter-1 (<- extracted from prototxt file)
      //      Run the next batch
      //          (Notes: 1. Wrap around training set when done
      //                  2. Batch size is extracted from protoxt file)
      //        Compute forward pass (iterate through vector of Bridge pointers)
      //        Compute backward pass (iterate through vector of Bridge
      //                               pointers backwards)
      //
      static void load_and_train_network(const char * file, const string input_model_file,
            const string output_model_file, bool time_iterations = false) {
        DeepNetConfig::train_ = true;

        BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
        Corpus * const corpus = DeepNet::load_network(file, solver_param, net_param, bridges, true);

        // Now, the bridges vector is filled. Check if we want to load weights.
        if (input_model_file != "NA") {
          read_model_from_file(bridges, input_model_file);
          std::cout << "Reading saved model from " << input_model_file << "\n";
        } else {
          std::cout << "Training new model\n";
        }
        
        // If we are going to run a validation set during training, also need to load it
        Corpus * val_corpus = NULL;
        if (solver_param.test_interval() > 0 && 
            solver_param.test_interval() <= solver_param.max_iter())
        {
            val_corpus = DeepNet::read_corpus_from_lmdb(net_param, false);
            assert(val_corpus->n_rows          == corpus->n_rows);
            assert(val_corpus->n_cols          == corpus->n_cols);
            assert(val_corpus->dim             == corpus->dim);
            if (val_corpus->mini_batch_size != corpus->mini_batch_size) {
                std::cout << "\nError: For now, the train and test sets must have matching batch sizes\n";
                std::cout << "Please update the train/test prototxt to make the batch sizes match\n";
                std::cout << "(there is no real need for this and it will be fixed in future releases)\n";
                exit(0);
            }
        }
        
        // Determine the snapshot name. By default this is the same as the ouput
        // model file (plus a timestamp)
        std::string snapshot_name = output_model_file;
        // If a snapshot name was specified in protobuf, use that instead of the 
        // output model name as the snapshot base name
        std::string snapshot_prefix = solver_param.snapshot_prefix();
        if (snapshot_prefix.length()) {
            snapshot_name = snapshot_prefix;
        }
        
        train_network(bridges, *corpus, net_param, solver_param, input_model_file, snapshot_name, *val_corpus, time_iterations);
        std::string output_model_name;
        if (output_model_file == "NA") {
          output_model_name = "trained_model.bin";
        } else {
          output_model_name = output_model_file;
        }
        // Write to file unless snapshot_after_train was set to false
        if (solver_param.snapshot_after_train()) {
          write_model_to_file(bridges, output_model_name);
          std::cout << "\nTrained model written to " + output_model_name +  ". Load it using the -input-model or -i flag.\n";
        } else {
          std::cout << "\nNot writing trained model to file (snapshot_after_train = false)\n";
        }
        
        // Step 4: Clean up
        clean_up(bridges, corpus);
      }

      static float load_and_test_network(const char * file, const string input_model_file, bool time_iterations = false) {
        DeepNetConfig::train_ = false;

        BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
        Corpus * const corpus = DeepNet::load_network(file, solver_param, net_param, bridges, false);

        if (input_model_file != "NA") {
          read_model_from_file(bridges, input_model_file);
          const float acc = test_network(bridges, *corpus, net_param, solver_param, time_iterations);
          clean_up(bridges, corpus);
          return acc;
        } else {
          std::cout << "No valid model file provided, use the -i or -input-model flag to specify your trained model." << std::endl;
          assert(false);
          return -1;
        }
      }
    };

// #include "DeepNet.hxx"

#endif
