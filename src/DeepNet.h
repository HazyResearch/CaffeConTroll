#ifndef _DeepNet_h
#define _DeepNet_h

#include <iostream>
#include <algorithm>
#include <boost/program_options.hpp>
#include <map>
#include <ctime>
#include "LogicalCube.h"
#include "Layer.h"
#include "Connector.h"
#include "Kernel.h"
#include "bridges/AbstractBridge.h"
#include "bridges/MaxPoolingBridge.h"
#include "bridges/AvePoolingBridge.h"
#include "bridges/ReLUBridge.h"
#include "bridges/ConvolutionBridge.h"
#include "bridges/FullyConnectedBridge.h"
#include "bridges/SoftmaxLossBridge.h"
#include "bridges/LRNBridge.h"
#include "bridges/ParallelizedBridge.h"
#include "bridges/DropoutBridge.h"
#include "bridges/ConcatBridge.h"
#include "bridges/GeneralConcatBridge.h"
#include "bridges/EltwiseBridge.h"
#include "bridges/SplitBridge.h"
#include "bridges/ScaleBridge.h"
#include "bridges/BatchNormBridge.h"
#include "parser/corpus.h"
#include "DeepNetConfig.h"
#include "util.h"

typedef LogicalCube<DataType_SFFloat, Layout_CRDB> LogicalCubeFloat;
typedef Layer<DataType_SFFloat, Layout_CRDB> LayerFloat;
typedef std::vector<LayerFloat *> LayerVec;
typedef AbstractBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver> Bridge;
typedef SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver> SoftmaxBridge;
typedef std::vector<Bridge *> BridgeVector;

// -----------------------------------------------------------------------
// Layer Info
// -----------------------------------------------------------------------
// This is only used within the function construct_network. It is used to
// map each layer to info for that layer in order to make the correct
// DAG connections. Once the network has been constructed there is no need 
// for the layer info, since it is contained in the bridges vector.
// 
// By default, only the layer is needed below, i.e. mapping the name in the
// proto to the output layer, so it can be used elsewhere in the DAG. All
// the other fields other than layer are needed to avoid unnecessary copies:
//
// When 2 PBridges are consecutive, e.g. a conv bridge is followed be a
// relu bridge, and if both bridges are on the GPU, then there is no need
// to copy back and forth from the host between bridges. This can be done
// currently in PBridge but the bridge needs to know:
//   1. Device information + data from the previous PBridge, and
//   2. Whether to share data with the next PBridge
// Since each PBridge is isolated from the others, those signals are 
// passed to the bridges and need to be stored for all predecessors,
// which is why we include them in the map below. The map below 
// stores device information for all previous bridges.
//
// SHADJIS TODO: But these are only defined for PBridges, not other types
// of bridges. All bridges in the loop below are just AbstractBridges, i.e.
// these vectors may not exist for that bridge. This is motivation to just
// merge AbstractBridge and ParallelizedBridge. So currently we only use the 
// vectors above to adjust PBridges, but should also for bridges which are not 
// part of a PBridge (softmax, concat, eltwise, split). Need to also make those
// pbridges, so that they too can avoid copies / extra allocations. Then,
// everything is a PBridge, so either merge PBridge and AbstractBridge or
// move certain scheduler elements to the AbstractBridge.
struct Layer_Info {
    LayerFloat * layer;
    // Each bridge except data has 1 top, so we use layer (the output of the bridge)
    // and bridge (the bridge itself) interchangably, but they are different objects
    // (the bridge transforms or "bridges" 1 layer to the next, the layer is data)
    Bridge * pbridge;
    size_t num_partitions_CPU;
    // Batch proportion per GPU
    std::vector<size_t> GPU_batch_sizes;
    // Which GPUs were used
    std::vector<int> gpu_to_device_id_map;
    // These vectors:
    //  - Allow GPU bridges (1 GPU or many) to share input data with the previous layer.
    //  - prev_data_cubes_higher is just the output layer of the bridge
    //    Higher means top (caffe) or output, and lower means bottom or input. It has nothing to do
    //    with conv lowering / lifting (should maybe rename "higher" to "top" or "out")
    //  - These cubes only are used for GPUs (1 or many) to share data, but they do not decide
    //    whether or not to share -- that is done by the GPU properties of the 2 bridges 
    //    being connected
    //  - the size of each vector is the # (CPU+GPU) partitions for the bridge
    std::vector< LogicalCubeFloat *> data_cubes_higher;
    std::vector< LogicalCubeFloat *> grad_cubes_higher;
    
    // Constructor
    Layer_Info() : layer(NULL), pbridge(NULL), num_partitions_CPU(0) {}
    
    // Utility methods
    size_t get_R() { assert(layer); return layer->p_data_cube->R; }
    size_t get_C() { assert(layer); return layer->p_data_cube->C; }
    size_t get_D() { assert(layer); return layer->p_data_cube->D; }
};

typedef std::map<std::string, Layer_Info> LayerMap;
typedef std::set<std::string> LayerSet;

class DeepNet {
  public:

    // Given an input dimension and the padding and stride (of a conv or pool), calculate the
    // output dimension
    static inline size_t compute_conv_next_layer_dimension(const size_t R_i, const size_t K,
        const size_t padding, const size_t stride ) {
      return (R_i + 2 * padding - K) / stride + 1;  // True for conv and pool
    }

    // Call this when done using the corpus and bridges
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
    
      // SHADJIS TODO: We currently hard-code that layer 0 is train and 
      // that if there is test mode that layer 1 is test
      // Note: This is different that the enum for train = phase 0 and test = phase 1
      // Recall cnn.proto:
      //
      //    enum Phase {
      //       TRAIN = 0;
      //       TEST = 1;
      //    }
      //
      // However in addition we also assume the prototxt layer 0 is train and if there
      // is test mode then layer 1 is test
      
      if (train) {
        const cnn::LayerParameter layer_param = net_param.layer(0); // SHADJIS TODO: Don't hard code train = layer 0
        string layer_type = layer_param.type();
        std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::toupper);
        if (layer_type == "DATA") {
          if (layer_param.include(0).phase() == 0) { // training phase
            return new Corpus(layer_param);
          }
        }
      } else {
        const cnn::LayerParameter layer_param = net_param.layer(1); // SHADJIS TODO: Don't hard code test = layer 1
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
        // SHADJIS TODO: We can generalize this later. Currently, pbridges have a parameter
        // 'num_bridges_in_this_group', which is the same for all bridges in that group.
        // Bridges in a group are executed in parallel, not sequentially. Currently this is
        // used for model parallelism, e.g. we make 4 bridges each with 1/4 depth and then
        // execute them in parallel on a different GPU. This can be generalized for any DAG.
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
        // if we run some in parallel, so use a while loop (see comment
        // above in run_forward_pass)
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

    static void write_full_snapshot(const BridgeVector bridges, const string base_filename, const int iter) {

        // This code also appends a time string
        //time_t rawtime;
        //struct tm * timeinfo;
        //char buffer[80];
        //time (&rawtime);
        //timeinfo = localtime(&rawtime);
        //strftime(buffer,80,"%d-%m-%Y-%I-%M-%S",timeinfo);
        //std::string str(buffer);
        std::string snapshot_name_model   = base_filename + ".snapshot_iter" + std::to_string(iter) + /*"." + str +*/ ".MODEL.bin";
        std::string snapshot_name_history = base_filename + ".snapshot_iter" + std::to_string(iter) + /*"." + str +*/ ".HISTORY.bin";
        std::string snapshot_name_iter    = base_filename + ".snapshot_iter" + std::to_string(iter) + /*"." + str +*/ ".ITER.bin";

        // Write model
        DeepNet::write_model_to_file(bridges, snapshot_name_model);

        // Write gradient history
        DeepNet::write_momentum_to_file(bridges, snapshot_name_history);

        // Write iteration #
        FILE * pFile = fopen (snapshot_name_iter.c_str(), "wb");
        if (!pFile)
          throw std::runtime_error("Error opening " + snapshot_name_iter);
        fwrite(&iter, sizeof(int), 1, pFile);
        fclose(pFile);
        std::cout << "======= Writing snapshot " << snapshot_name_model << " =======" << std::endl;
    }

    static void read_full_snapshot(BridgeVector & bridges, const string base_filename) {
        // Open these files: E.g.
        //   alexnet_solver.prototxt.snapshot_iter500.MODEL.bin
        //   alexnet_solver.prototxt.snapshot_iter500.HISTORY.bin
        //   alexnet_solver.prototxt.snapshot_iter500.ITER.bin
        read_model_from_file(bridges, base_filename + ".MODEL.bin");
        read_momentum_from_file(bridges, base_filename + ".HISTORY.bin");

        // Also read iteration number
        // SHADJIS TODO: To start exactly need to also move fw in dataset
        // to this iteration
        FILE * pFile;
        pFile = fopen ((base_filename + ".ITER.bin").c_str(), "rb");
        int iter;
        size_t num_elements_read = fread(&iter, sizeof(int), 1, pFile);
        assert(num_elements_read == 1);
        LogicalCubeFloat * model;
        LogicalCubeFloat * bias;
        for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
          model = (*bridge)->get_model_cube();
          if (model) {
            (*bridge)->set_current_iter(iter);
          }
          bias = (*bridge)->get_bias_cube();
          if (bias) {
            (*bridge)->set_current_iter(iter);
          }
        }
        fclose(pFile);

        std::cout << "Read snapshot " << base_filename << std::endl;
    }

    // Write the models of all bridges to a file
    static void write_model_to_file(const BridgeVector bridges, const string model_file) {
      FILE * pFile = fopen (model_file.c_str(), "wb");
      if (!pFile)
        throw std::runtime_error("Error opening " + model_file);

      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
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
      pFile = fopen (model_file.c_str(), "rb"); // SHADJIS TODO: if (!pFile') error
      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
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

    // Write the gradient history of all bridges to a file
    static void write_momentum_to_file(const BridgeVector bridges, const string momentum_file) {
      FILE * pFile = fopen (momentum_file.c_str(), "wb");
      if (!pFile)
        throw std::runtime_error("Error opening " + momentum_file);

      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        model = (*bridge)->get_model_cube();
        if (model) {
          (*bridge)->force_device_to_host_model_history_copy();
          fwrite((*bridge)->get_model_history_host_ptr(), sizeof(DataType_SFFloat), model->n_elements, pFile);
        }
        bias = (*bridge)->get_bias_cube();
        if (bias) {
          (*bridge)->force_device_to_host_bias_history_copy();
          fwrite((*bridge)->get_bias_history_host_ptr(), sizeof(DataType_SFFloat), bias->n_elements, pFile);
        }
      }
      fclose(pFile);
    }

    // Read in the gradient history of all bridges from a file
    static void read_momentum_from_file(BridgeVector & bridges, const string model_file) {
      FILE * pFile;
      pFile = fopen (model_file.c_str(), "rb");
      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        model = (*bridge)->get_model_cube();
        if (model) {
          size_t num_elements_read = fread((*bridge)->get_model_history_host_ptr(), sizeof(DataType_SFFloat), model->n_elements, pFile);
          assert(num_elements_read == model->n_elements);
          (*bridge)->force_host_to_device_model_history_copy();
        }
        bias = (*bridge)->get_bias_cube();
        if (bias) {
          size_t num_elements_read = fread((*bridge)->get_bias_history_host_ptr(), sizeof(DataType_SFFloat), bias->n_elements, pFile);
          assert(num_elements_read == bias->n_elements);
          (*bridge)->force_host_to_device_bias_history_copy();
        }
      }
      fclose(pFile);
    }

    // Get the total number of parameters (bias + model) in all bridges of the network
    static size_t get_parameter_size(const BridgeVector bridges) {

      size_t total_size = 0;
      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
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

    // Get the total number of bridges containing parameters
    static size_t get_num_model_bridges(const BridgeVector bridges) {
      size_t num_model_bridges = 0;
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        if ( (*bridge)->get_model_cube() || (*bridge)->get_bias_cube() ) {
          num_model_bridges += 1;
        }
      }
      return num_model_bridges;
    }

    // Given a buffer (already allocated on the host), fill it with all the gradients
    // To know how big to make this buffer see get_parameter_size()
    static void get_all_gradients(const BridgeVector bridges, DataType_SFFloat * buffer) {  

      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
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

    // Given a buffer (already allocated on the host), fill it with the ith gradients
    // To know how big to make this buffer see get_parameter_size()
    static int get_ith_gradient(const BridgeVector bridges, DataType_SFFloat * buffer, int i) {  

      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
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
      LogicalCubeFloat * model;
      model = (bridges[i])->get_model_cube();
      if (model) {
        memcpy(buffer, (bridges[i])->get_model_gradient_host(), sizeof(DataType_SFFloat) * model->n_elements);
      }
    }
    static void get_ith_gradient_bias_only(const BridgeVector bridges, DataType_SFFloat * buffer, int i) {  
      LogicalCubeFloat * bias;
      bias = (bridges[i])->get_bias_cube();
      if (bias) {
        memcpy(buffer, (bridges[i])->get_bias_gradient_host(),  sizeof(DataType_SFFloat) * bias->n_elements);
      }
    }

    // Given a buffer of all the gradients in the network, update all the models of all the bridges
    static void update_all_models_with_gradients(const BridgeVector bridges, DataType_SFFloat * gradients_concatenated) {

      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
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

      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
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

      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
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

      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;

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

      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;

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
      LogicalCubeFloat * model;
      model = bridges[i]->get_model_cube();
      if(model){
        memcpy(model->get_p_data(), models_concatenated, sizeof(DataType_SFFloat) * model->n_elements);
        bridges[i]->force_host_to_device_model_copy();
      }
    }
    static void set_ith_bias_only(const BridgeVector bridges, DataType_SFFloat * models_concatenated, int i) {  
      LogicalCubeFloat * bias;
      bias = bridges[i]->get_bias_cube();
      if (bias) {
        memcpy(bias->get_p_data(), models_concatenated, sizeof(DataType_SFFloat) * bias->n_elements);
        bridges[i]->force_host_to_device_bias_copy();
      }
    }

    // Like read_model_from_file() but read model from a memory buffer
    static void set_all_models(const BridgeVector bridges, DataType_SFFloat * models_concatenated) {  

      LogicalCubeFloat * model;
      LogicalCubeFloat * bias;
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
    
    
    // Return hw concurrency, which is usually #vcpu
    // For now use powers of 2, can change this later
    static unsigned int get_hw_concurrency() {
      unsigned int hw_concurrency = 1;
      while (true) {
        if (hw_concurrency*2 > std::thread::hardware_concurrency()) {
            break;
        }
        hw_concurrency *= 2;
      }
      return hw_concurrency;
    }
    

    static void get_model_parallelism_info(std::vector <float> & GPU_depth_proportions, std::vector <int> & GPUs_used_for_model_parallelism,
            const cnn::LayerParameter & layer_param, const size_t output_D) {

        GPU_depth_proportions.push_back(layer_param.gpu_0_depth_proportion());
        GPU_depth_proportions.push_back(layer_param.gpu_1_depth_proportion());
        GPU_depth_proportions.push_back(layer_param.gpu_2_depth_proportion());
        GPU_depth_proportions.push_back(layer_param.gpu_3_depth_proportion());
        float portion_of_depth_per_GPU = 0.;   // Eventually will not be needed, once depth partitions can be nonuniform
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
          assert(GPUs_used_for_model_parallelism.size() > 1);
        }
        // Otherwise no model parallelism on GPU (could still be using any # of GPUs for data parallelism,
        // but that is for the pbridge to handle)
        else {
          assert(portion_of_depth_per_GPU == 0);
          assert(GPUs_used_for_model_parallelism.size() == 0);
        }
    }

    
    // Return the name of the first data layer and its first top
    // Maybe in the future we can handle multiple data layers and tops but for now we 
    // will just handle the first layer and its first top
    static std::string get_data_layer_name(const cnn::NetParameter & net_param) {

      for (int i_layer = 0; i_layer < net_param.layer_size(); ++i_layer) {

        const cnn::LayerParameter layer_param = net_param.layer(i_layer);
        string layer_type = layer_param.type();
        std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::toupper);
        if (layer_type == "DATA")
        {
          if (layer_param.top(0).c_str() != layer_param.name()) {
            std::cout << "Error, currently the data top(0) name must match data layer name" << std::endl;
            assert(false);
          }
          return layer_param.name();
        }
      }
      std::cout << "Error, currently a data layer is needed" << std::endl;
      assert(false);
      return "";
    }
    
    
    // Get the next bridge which has all its bottoms complete
    //  - skip the layer if it is a data layer (no bottoms), accuracy layer (2 bottoms)
    //  - skip the layer if it is in our set of layers (already done)
    //  - Assert each bridge has 1 top and its name is the top name (for data 2 tops is ok but we ignored one. All but data have 1 top),
    //    or that its bottom and its top are the same
    //  - return 1st one with all its bottoms already processed        
    //  - If it is loss, just check if either bottom is done
    //  - Multiple bottoms is only true for concat, elementwise, and 
    static size_t get_next_bridge_id(const LayerMap & bridge_name_to_info, const cnn::NetParameter & net_param, bool & all_bridges_done) {
      
        all_bridges_done = true;
        for (int i_layer = 0; i_layer < net_param.layer_size(); ++i_layer) {
        
            const cnn::LayerParameter layer_param = net_param.layer(i_layer);
            string layer_type = layer_param.type();
            std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::toupper);
            string layer_name = layer_param.name();
            
            // skip the layer if it is a data layer (no bottoms), accuracy layer (handled automatically),
            // or if we already processed it
            if (layer_type == "DATA" || layer_type == "ACCURACY" || bridge_name_to_info.count(layer_name)) {
                continue;
            }
            
            // We have found a bridge which has not been processed yet
            all_bridges_done = false;
            
            // all bridges but data have 1 top
            assert(layer_param.top_size() == 1);
            // all bridges also either should be named their top, or have their bottom be their top
            // (e.g. ReLU should have same top and bottom, but that is not the name of the ReLU)
            if (layer_param.top(0) != layer_name) {
                assert(layer_param.bottom_size() == 1);            
                assert(layer_param.bottom(0) == layer_param.top(0));
            }
            
            // For bridges with more than 1 bottom, assert it is concat, elementwise or loss
            assert(layer_param.bottom_size() > 0);
            if (layer_param.bottom_size() > 1) {
                assert(layer_type == "CONCAT" || layer_type == "ELTWISE" || layer_type == "SOFTMAXWITHLOSS" || layer_type == "SOFTMAX");
            }
            
            // Now iterate over all bottoms and return this bridge if all bottoms have been processed
            // If not, move to the next bridge
            
            // For loss bridges, only need one bottom to be done (since we ignore labels)
            // Handle this case separately
            if (layer_type == "SOFTMAXWITHLOSS" || layer_type == "SOFTMAX") {
                for (int i_bottom = 0; i_bottom < layer_param.bottom_size(); ++i_bottom) {
                    // If even one of these match, then this layer is ready to process
                    if (bridge_name_to_info.count(layer_param.bottom(i_bottom))) {
                        return i_layer;
                    }
                }
            }
            // Otherwise every bottom needs to be done
            else {
                bool all_bottoms_done = true;
                for (int i_bottom = 0; i_bottom < layer_param.bottom_size(); ++i_bottom) {
                    // If even one of these match, then this layer is ready to process
                    if (!bridge_name_to_info.count(layer_param.bottom(i_bottom))) {
                        all_bottoms_done = false;
                        break;
                    }
                }
                if (all_bottoms_done) {
                    return i_layer;
                }
            }
        }
        
        return -1;
    }
        
        
    // Count the number of times this layer is used as the bottom to other layers.
    // Normally we would split if there is more than 1 use (with some exceptions, e.g.
    // accuracy does not count as a use).
    // However, before adding the split we need to first add all in-place operations.
    //
    // Example: if there is conv -> relu -> pool, the conv will have 2 uses, and relu
    // will have 0 (because relu is in-place so according to caffe syntax its top and
    // bottom are both conv). However we do not want to return true for the conv, only
    // for the relu.
    static int should_insert_split_bridge(const LayerMap & bridge_name_to_info, std::string current_top_name,
        const cnn::NetParameter & net_param, int & num_uses) {

      for (int i_layer = 0; i_layer < net_param.layer_size(); ++i_layer) {
        const cnn::LayerParameter layer_param = net_param.layer(i_layer);
        
        // Ignore certain layers
        string layer_type = layer_param.type();
        std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::toupper);
        if (layer_type == "ACCURACY") {
            continue;
        }
        
        // Iterate over all bottoms
        bool bottom_matches = false;
        for (int i_bottom = 0; i_bottom < layer_param.bottom_size(); ++i_bottom) {
            if (layer_param.bottom(i_bottom) == current_top_name) {
                bottom_matches = true;
                break;
            }
        }
        if (bottom_matches) {
            // One of the bottoms match, however we don't want to count uses if it is an
            // in-place player. Moreover, if that in-place layer has not been processed yet,
            // return false since we want to apply them all first.
            assert(layer_param.top_size() == 1);    // True for all layers but data which have no bottom
            if (current_top_name == layer_param.top(0)) {
                // This is an in-place bridge. If it hasn't been processed yet, return false for now
                // and we'll insert the split after it gets processed
                if (!bridge_name_to_info.count(layer_param.name())) {
                    return false;
                }
            }
            // Otherwise this is not an in-place bridge, so count it as a use
            else {
                num_uses += 1;
            }
        }
      }
      return num_uses > 1;
    }
    
    
    static std::string get_split_name(std::string input_name_to_split, int split_index) {
        return input_name_to_split + "__SPLIT_" + std::to_string(split_index) + "_";
    }


    static std::string get_full_name(std::string layer_name, std::map<std::string, int> split_layer_to_use_count) {
        // not storing and re-using find iterator because map size is probably 2
        if (split_layer_to_use_count.count(layer_name)) {
            return get_split_name(layer_name, split_layer_to_use_count[layer_name]);
        }
        return layer_name;
    }


    static void increment_use(std::string layer_name, std::map<std::string, int> & split_layer_to_use_count) {
        // not storing and re-using find iterator because map size is probably 2
        if (split_layer_to_use_count.count(layer_name)) {
            split_layer_to_use_count[layer_name] += 1;
        }
    }


    // This takes in the bridge vector (which has been initialized to be empty in load_and_train_network)
    // and builds up a list of bridges in the vector in the order in which they will be executed in the forward
    // pass. Only the bridges variable is modified.
    static void construct_network(BridgeVector & bridges, Corpus & corpus, const cnn::NetParameter & net_param,
        const cnn::SolverParameter & solver_param) {
      
      CPUDriver * const driver = new CPUDriver(); // SHADJIS TODO: delete this later or put on stack
      unsigned int hw_concurrency = get_hw_concurrency();
      unsigned int nphysical_cores = std::max(int(1), int(hw_concurrency/2));
      assert(hw_concurrency > 0);
      
      // Create maps which map layer names to important info for those layers
      LayerMap bridge_name_to_info;
      
      // Also create a set of layers which don't have any weight layers before them
      // These don't need back prop so we can skip the backwards pass and also avoid
      // adding split bridges.
      // SHADJIS TODO: Currently we only skip backwards data calculation for the first 
      // conv layer but this could be applied to all layers before it too, or the first 
      // fc and all layers before it (if the net has no conv). No backward pass wrt data 
      // is needed for the first fc or conv layer (whichever is first) as well as everything 
      // before it. This can also avoid adding a split bridge for model-parallel FC.
      LayerSet before_first_weight_layer;      

      // Finally, we need to keep track of which layers are split into multiple layers.
      // Below when we insert new split layers their name changes (see get_split_name),
      // but the proto will still assume and search for the un-split name. So use this
      // map to keep track of the layers which were split.
      // Note: alternatively this information could have been kept inside of Layer_Info,
      // i.e. rather than Layer_Info storing 1 layer, it would store multiple layers for
      // the split bridges. However this way keeps the two isolated, i.e. Layer_Info is
      // for one layer only.
      std::map<std::string, int> split_layer_to_use_count;
      
      // Create the Logical Cubes for the initial data layer
      std::string data_layer_name = get_data_layer_name(net_param);
      size_t data_R = corpus.n_rows, data_C = corpus.n_cols, data_D = corpus.dim, B = corpus.mini_batch_size;
      LogicalCubeFloat * input_data = new LogicalCubeFloat(corpus.images->physical_get_RCDslice(0), data_R, data_C, data_D, B);
      LogicalCubeFloat * input_grad = new LogicalCubeFloat(data_R, data_C, data_D, B);
      bridge_name_to_info[data_layer_name].layer = new LayerFloat(input_data, input_grad);
      before_first_weight_layer.insert(data_layer_name);
      
      // This data layer may be used in multiple places, but because it precedes all model
      // bridges back prop is not needed for it, so we do not need to insert a split bridge
      // For this first data layer, append a split bridge if it is used in other places
      
      // Now start adding each bridge. We add bridges when all their bottoms have been added to bridge_name_to_info.
      while (true) {
      
        // Find the next bridge whose predecessor layers ('bottom' dependencies) have all been processed already
        bool all_bridges_done = false;
        // SHADJIS TODO: Could probably just return the next in order, i.e. the layers usually
        // appear such that a layer comes after all its bottoms
        size_t next_bridge_id = get_next_bridge_id(bridge_name_to_info, net_param, all_bridges_done);
        if (all_bridges_done) {
          break;
        } else {
          assert(next_bridge_id >= 0); // If all bridges aren't done, a bridge must have been found
        }
        const cnn::LayerParameter layer_param = net_param.layer(next_bridge_id);
        string layer_type = layer_param.type();
        std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::toupper);
        std::string layer_name = layer_param.name();
       
        // Some temp variables which will be used below
        Bridge * new_bridge = NULL;
        LogicalCubeFloat * new_data = NULL;
        LogicalCubeFloat * new_grad = NULL;
        LayerFloat * new_layer = NULL;

        // SHADJIS TODO: Some layers, e.g. ReLU and dropout, have the same top
        // and bottom layers and therefore don't need to allocate any output cubes.
        // For now, we will avoid this allocation only in PBridge (since it matters
        // more for the GPU), i.e. just for ReLU/dropout. When we support others
        // (e.g. concat may also have same top/bottom) as PBridges, then those will
        // also benefit from not needing extra cube allocations (and also not needing
        // copies to/from host if entire network is on GPU)
        // SHADJIS TODO: do this if possible for eltwise, split, concat
        // Note the map is unaffected since we still make a new layer by its name,
        // i.e. we treat this as a new layer and the only difference is 
        //   1. setting this bool to save memory/copies, and
        //   2. these top == bottom bridges are not counted when counting fan-out
        //      (to create split bridges, see below)
        //
        // Note: setting this only makes 1 change to PBridge, which is to join the
        // pointers of the output layer to the input layer on the GPU only.
        // Within the bridges, ReLU currently works with or without this on the GPU,
        // but dropout GPU asserts it to be true (it is not hard to make it work 
        // without this on the GPU).
        // 
        bool share_input_output_layer = false;
        if (layer_param.top_size() == 1 && layer_param.bottom_size() == 1 &&
            layer_param.top(0) == layer_param.bottom(0))
        {
            share_input_output_layer = true;
        }

        // Make the layer info (the bridge, the device info, the output layer) and add it to the map
        // Also add the bridge to the bridges vector
        // SHADJIS TODO: This if/else for each bridge type can be refactored to make it shorter
        if (layer_type == "CONVOLUTION")
        {
            std::cout << "Constructing CONV bridge" << std::endl;
            
            // Get the previous bridge's information
            assert(layer_param.bottom_size() == 1);
            std::string bottom_name = layer_param.bottom(0);
            Layer_Info bottom_info = bridge_name_to_info[get_full_name(bottom_name, split_layer_to_use_count)];
            
            // Read the settings specific to this bridge type
            if (layer_param.convolution_param().group() != 1) {
                std::cout << "    Warning: grouping is no longer supported and will be ignored. You can do this using a DAG instead." << std::endl;
            }
            const size_t K = layer_param.convolution_param().kernel_size();
            const size_t padding = layer_param.convolution_param().pad();
            const size_t stride = layer_param.convolution_param().stride();
            size_t output_R = compute_conv_next_layer_dimension(bottom_info.get_R(), K, padding, stride);
            size_t output_C = compute_conv_next_layer_dimension(bottom_info.get_C(), K, padding, stride);
            size_t output_D = layer_param.convolution_param().num_output();

            // Create the new layer
            //
            // SHADJIS TODO: Also, if the 2 bridges are sharing device data pointers,
            // then there is no need to copy the data to or from the host, so these
            // cube allocations (the cubes own their own data) are unnecessary since
            // those cubes are never used anywhere in pbridge. This won't be true
            // for conv but could be true for the other layers below, e.g. ReLU
            // Specifically, for those in-place bridges (e.g. ReLU, Dropout, Scale, Batch Norm)
            // can skip the allocation below, and instead set the pointers to those in the
            // input layer.
            new_data = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_layer = new LayerFloat(new_data, new_grad);

            // Create the new bridge
            //
            // SHADJIS TODO: Can make these input args, e.g. like gpu allocation, read # cores to use from file
            new_bridge = new ParallelizedBridge<DataType_SFFloat, ConvolutionBridge>
                     (bottom_info.layer, new_layer, &layer_param, &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1, // TODO: need a CMD line option here -- but currently we do not have the interface to do that.
                     bottom_info.num_partitions_CPU, bottom_info.GPU_batch_sizes, bottom_info.gpu_to_device_id_map, bottom_info.data_cubes_higher, bottom_info.grad_cubes_higher,
                     share_input_output_layer);
            
            new_bridge->name = layer_name;
            if (before_first_weight_layer.count(bottom_name)) {
                new_bridge->needs_to_calc_backward_grad = false;
            }
            bridges.push_back(new_bridge);
            
            // -------------- Scheduler Update ----------------
            // SHADJIS TODO: Can refactor this, it is the same for most bridges (except the DAG ones like
            // concat, split, eltwise)
            bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
            if (bridge_shares_data_with_prev_bridge) {
              if (bottom_info.pbridge) {
                bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
              }
            }
            bridge_name_to_info[layer_name].layer = new_layer;
            bridge_name_to_info[layer_name].pbridge = new_bridge;
            bridge_name_to_info[layer_name].GPU_batch_sizes = new_bridge->get_GPU_batch_sizes();
            bridge_name_to_info[layer_name].num_partitions_CPU = new_bridge->get_num_partitions_CPU();
            bridge_name_to_info[layer_name].gpu_to_device_id_map = new_bridge->get_used_gpu_to_device_id_map();
            bridge_name_to_info[layer_name].data_cubes_higher = new_bridge->get_data_cubes_higher();
            bridge_name_to_info[layer_name].grad_cubes_higher = new_bridge->get_grad_cubes_higher();
            increment_use(bottom_name, split_layer_to_use_count); // If this was a split, use another split next time
            // ----------End of Scheduler Update --------------
        }
        else if (layer_type == "INNERPRODUCT")
        {
            size_t output_R = 1;
            size_t output_C = 1;
            size_t output_D = layer_param.inner_product_param().num_output();
            
            // FC layers can use model parallelism across GPUs
            // Check if this FC layer uses model parallelism
            std::vector <float> GPU_depth_proportions;  // Fraction of model per GPU 0 1 2 3
            std::vector <int> GPUs_used_for_model_parallelism;  // GPUs to use (e.g. 0,1), redundant given GPU_depth_proportions but easier to read
            get_model_parallelism_info(GPU_depth_proportions, GPUs_used_for_model_parallelism, layer_param, output_D);
            // Note if number_of_model_parallel_groups = 0 then GPUs can still be used for data parallelism,
            // but this just means that there will not be multi-GPU model parallelism
            int number_of_model_parallel_groups = GPUs_used_for_model_parallelism.size();
            
            // Get the previous bridge's information
            assert(layer_param.bottom_size() == 1);
            std::string bottom_name = layer_param.bottom(0);
            Layer_Info bottom_info = bridge_name_to_info[get_full_name(bottom_name, split_layer_to_use_count)];
            
            // Now create the bridges
            
            // Normal case: No model parallelism, so just create a single pbridge. This is the normal case.
            if (number_of_model_parallel_groups == 0) {
            
              std::cout << "Constructing FC bridge" << std::endl;
            
              // Create the new layer
              new_data = new LogicalCubeFloat(output_R, output_C, output_D, B);
              new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
              new_layer = new LayerFloat(new_data, new_grad);
     
              // Create the new bridge
              new_bridge = new ParallelizedBridge<DataType_SFFloat, FullyConnectedBridge>
                       // using hw_concurrency / 2 since GEMM faster with #physical
                       (bottom_info.layer, new_layer, &layer_param, &solver_param, driver, min<size_t>(1, corpus.mini_batch_size), nphysical_cores,
                       bottom_info.num_partitions_CPU, bottom_info.GPU_batch_sizes, bottom_info.gpu_to_device_id_map, bottom_info.data_cubes_higher, bottom_info.grad_cubes_higher,
                       share_input_output_layer);
            
              new_bridge->name = layer_name;
              if (before_first_weight_layer.count(bottom_name)) {
                  // new_bridge->needs_to_calc_backward_grad = false;
              }
              bridges.push_back(new_bridge);
              
              // -------------- Scheduler Update ----------------
              bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
              if (bridge_shares_data_with_prev_bridge) {
                if (bottom_info.pbridge) {
                  bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
                }
              }
              bridge_name_to_info[layer_name].layer = new_layer;
              bridge_name_to_info[layer_name].pbridge = new_bridge;
              bridge_name_to_info[layer_name].GPU_batch_sizes = new_bridge->get_GPU_batch_sizes();
              bridge_name_to_info[layer_name].num_partitions_CPU = new_bridge->get_num_partitions_CPU();
              bridge_name_to_info[layer_name].gpu_to_device_id_map = new_bridge->get_used_gpu_to_device_id_map();
              bridge_name_to_info[layer_name].data_cubes_higher = new_bridge->get_data_cubes_higher();
              bridge_name_to_info[layer_name].grad_cubes_higher = new_bridge->get_grad_cubes_higher();
              // ----------End of Scheduler Update --------------
            }
            
            // =============================================================================================================
            // MODEL PARALLELISM
            // -------------------------------------------------------------------------------------------------------------
            // Now we need to create multiple model-parallel bridges
            // For now, we know this will use GPUs (CPU case is handled above, and is same as data parallelism for CPUs)
            // Also, we assume that the data is on the host, i.e. do not share with previous bridge
            // 
            // Normally we would create a single FC parallelized bridge. This already handles data parallelism, and could also
            // internally handle model parallelism. However, model parallelism can also be implemented as a DAG, by adding a 
            // split, then multiple FC layers, then a concat. Since we already support DAGs, this method is simpler than
            // adding model parallelism within a pbridge (therefore pbridge could be renamed DataParallelizedBridge). The
            // one change from a regular DAG is that it is necessary to run the bridges in parallel. This is done using 
            // threads inside run_forward_pass and run_backward_pass. 
            // 
            // The alternative would be to implement model parallelism inside a pbridge, and then just call forward/backward
            // normally on the bridge. Internally the pbridge would partition by model instead of data. A description of the
            // changes needed to implement this are here:
            // https://github.com/HazyResearch/CaffeConTroll/blob/3ba31e9241986fc2f306a34b6f55546702bf437f/src/DeepNet.h#L1058
            // =============================================================================================================
            else {
              // -----------------------------------------------------------------------------------------------------------
              // Add a Split Bridge
              // -----------------------------------------------------------------------------------------------------------
              // The split bridge is on the CPU. We will always do a copy to/from CPU before model parallelism because 
              // model parallelism only makes sense on > 1 GPU and we need all the data on that GPU. Since all the data
              // never exists on multiple GPUs at the same time (because we use data parallelism), we need a copy of the
              // data here (even if we had a concat just now because there was grouping before, we need both that concat
              // and this split bridge since we need to merge all the data before doing model parallelism)
              // See all the comments in the general split bridge below for more information.
              std::cout << "Constructing SPLIT bridge for upcoming model-parallel FC. Splitting depth from 1 to " 
                        << number_of_model_parallel_groups << " partitions" << std::endl;
              
              // Note: Similarly to how concat never uses the input layer, split never uses the output layer
              // Therefore we can just make empty cubes. I don't know if the sizes matter
              size_t input_R = bottom_info.get_R();
              size_t input_C = bottom_info.get_C();
              size_t input_D = bottom_info.get_D();
              new_data = new LogicalCubeFloat(NULL, input_R, input_C, input_D, B);
              new_grad = new LogicalCubeFloat(NULL, input_R, input_C, input_D, B);
              new_layer = new LayerFloat(new_data, new_grad);
              
              // The input layer is the layer from the previous bridge. We copy that directly to each output cube in 
              // the fw pass (or just share the pointers)
              LayerFloat * input_layer_of_split_bridge = bottom_info.layer;
              new_bridge = new SplitBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>(input_layer_of_split_bridge,
                  new_layer, &layer_param, &solver_param, driver);
                  
              // See comment in general split bridge below: now we need to create output layers, one per bridge in the upcoming 
              // model-parallel group.
              LayerVec split_output_layers;
              for (int i = 0; i < number_of_model_parallel_groups; ++i) {
                // Do not allocate
                new_data = new LogicalCubeFloat(input_layer_of_split_bridge->p_data_cube->get_p_data(), input_R, input_C, input_D, B);
                // Allocate
                new_grad = new LogicalCubeFloat(input_R, input_C, input_D, B);
                new_layer = new LayerFloat(new_data, new_grad);
                ((SplitBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>*)new_bridge)->p_output_layers.push_back(new_layer);
                split_output_layers.push_back(new_layer);
              }
              assert(split_output_layers.size() == size_t(number_of_model_parallel_groups));
              new_bridge->name = "SPLIT_" + layer_name;    // This name is arbitrary and does not matter
              bridges.push_back(new_bridge);
              
              // Note that we have added this split bridge to the bridges vector but have not yet added anything to our
              // LayerMap. This is because we only need the final top layer, and that will come from the concat later.

              // -----------------------------------------------------------------------------------------------------------
              // Create each bridge in the model parallel FC bridge
              // -----------------------------------------------------------------------------------------------------------
              std::cout << "Constructing Model-Parallel FC bridge with Total Depth = " << output_D << std::endl;
              
              // Like all pbridges we will need to pass in the device info, however the split bridge we just added has no device info
              // for its output layer. We could add that split bridge output layer to the LayerMap, but since it is on the host
              // we can just use an empty Layer_Info instead.
              Layer_Info empty_layer_info;
              LayerVec model_parallel_fc_output_layers;
              for (int i = 0; i < number_of_model_parallel_groups; i++) {
              
                // Read the depth for this bridge
                int gpu_idx = GPUs_used_for_model_parallelism[i];
                int output_D_partition = output_D * GPU_depth_proportions[gpu_idx];
                assert(GPU_depth_proportions[gpu_idx] > 0);
                std::cout << "  Constructing partition of model-parallel FC bridge with partial depth = " << output_D_partition << std::endl;
                
                // ---------------------------------------------------------------------------------------------------------
                // Update the solver to only use this GPU, if this is a GPU bridge
                // ---------------------------------------------------------------------------------------------------------
                // SHADJIS TODO: Fix hack: This is a hack now, I am going to make a new layer_param object and
                // change the GPU allocations. This is a hack and could be fixed if we do model parallelism inside the pbridge.
                // Also for now this has a small memory leak
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
                // Also make a solver_param_tmp if random_seed is set which assigns a different seed to each bridge
                // SHADJIS TODO: I noticed no improvement in statistical efficiency with this, but can test more later
                //cnn::SolverParameter * solver_param_tmp = new cnn::SolverParameter(solver_param); // Copy constructor
                //if (solver_param_tmp->random_seed() != -1) {
                //  solver_param_tmp->set_random_seed(solver_param_tmp->random_seed() + gpu_idx);
                //}
                
                // ---------------------------------------------------------------------------------------------------------
                // Now create the bridge for this group
                // ---------------------------------------------------------------------------------------------------------
                new_data = new LogicalCubeFloat(output_R, output_C, output_D_partition, B);
                new_grad = new LogicalCubeFloat(output_R, output_C, output_D_partition, B);
                new_layer = new LayerFloat(new_data, new_grad);

                // SHADJIS TODO: Can also use a new driver for each, i.e. rather than driver, pass in new CPUDriver().
                // This didn't cause any problems but since we will run these pbridges in parallel, using the same driver
                // means that the drivers' internal class variables will be shared. Currently drivers have no variables but they may later.
                new_bridge = new ParallelizedBridge<DataType_SFFloat, FullyConnectedBridge>
                         // using hw_concurrency / 2 since GEMM faster with #physical
                         (split_output_layers[i], new_layer, layer_param_tmp, &solver_param, driver, min<size_t>(1, corpus.mini_batch_size), nphysical_cores,
                         empty_layer_info.num_partitions_CPU, empty_layer_info.GPU_batch_sizes, empty_layer_info.gpu_to_device_id_map, empty_layer_info.data_cubes_higher, empty_layer_info.grad_cubes_higher,
                         share_input_output_layer);

                new_bridge->name = layer_name + "_" + std::to_string(i);
                if (before_first_weight_layer.count(bottom_name)) {
                    // new_bridge->needs_to_calc_backward_grad = false;
                }
                bridges.push_back(new_bridge);
                model_parallel_fc_output_layers.push_back(new_layer);
                
                // Update this bridge to set its model parallelism group size
                new_bridge->set_model_parallelism_group_size(number_of_model_parallel_groups);
              }
              
              // -----------------------------------------------------------------------------------------------------------
              // We finished constructing our fc bridges
              // Now make a concat again to restore group size to 1
              // -----------------------------------------------------------------------------------------------------------
              std::cout << "Constructing CONCAT bridge to merge " << number_of_model_parallel_groups << " depth partitions of model-parallel FC bridge" << std::endl;
              new_data = new LogicalCubeFloat(output_R, output_C, output_D, B);
              new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
              new_layer = new LayerFloat(new_data, new_grad);
              // Note that now this concat gets model_parallel_fc_output_layers as input since it needs to merge the layers we just made above
              // Note: Concat never uses the input layer, i.e. we pass in model_parallel_fc_output_layers[0] but this p_input_layer is never used,
              // since instead we use p_input_layers which contains all of them
              new_bridge = new ConcatBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>(model_parallel_fc_output_layers[0],
                  new_layer, &layer_param, &solver_param, driver);
              for (int i = 0; i < number_of_model_parallel_groups; i++) {
                ((ConcatBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>*)new_bridge)->p_input_layers.push_back(model_parallel_fc_output_layers[i]);
              }
              new_bridge->name = "CONCAT_" + layer_name;    // This name is arbitrary and does not matter
              bridges.push_back(new_bridge);
              // There is no pbridge, but add this to the map anyway since it needs to go to other bridges
              // Important: note that the name used is the original layer name, even though the bridge is the concat
              bridge_name_to_info[layer_name].layer = new_layer;
            }
            increment_use(bottom_name, split_layer_to_use_count); // If this was a split, use another split next time
            
            // =============================================================================================================
            // End of Model Parallelism
            // -------------------------------------------------------------------------------------------------------------
            // - The rest of the implementation happens in run_forward/backward_pass
            // =============================================================================================================
        }
        else if (layer_type == "POOLING")
        {
            // Get the previous bridge's information
            assert(layer_param.bottom_size() == 1);
            std::string bottom_name = layer_param.bottom(0);
            Layer_Info bottom_info = bridge_name_to_info[get_full_name(bottom_name, split_layer_to_use_count)];
            
            // Read the settings specific to this bridge type
            const size_t K = layer_param.pooling_param().kernel_size(), stride = layer_param.pooling_param().stride();
            size_t output_R = compute_conv_next_layer_dimension(bottom_info.get_R(), K, 0, stride);
            size_t output_C = compute_conv_next_layer_dimension(bottom_info.get_C(), K, 0, stride);
            size_t output_D = bottom_info.get_D();

            // Create the new layer
            // input_D same as output_D
            new_data = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_layer = new LayerFloat(new_data, new_grad);

            // Create the new bridge
            // Recall cnn.proto:
            //
            //  enum PoolMethod {
            //    MAX = 0;
            //    AVE = 1;
            //  }
        
            // Read the settings specific to this bridge type
            if (layer_param.pooling_param().pool() == 0) {
                std::cout << "Constructing MAXPOOLING bridge" << std::endl;
                new_bridge = new ParallelizedBridge<DataType_SFFloat, MaxPoolingBridge>(bottom_info.layer, new_layer, &layer_param,
                       &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1,
                       bottom_info.num_partitions_CPU, bottom_info.GPU_batch_sizes, bottom_info.gpu_to_device_id_map, bottom_info.data_cubes_higher, bottom_info.grad_cubes_higher,
                       share_input_output_layer);
            } else if (layer_param.pooling_param().pool() == 1) {
                std::cout << "Constructing AVEPOOLING bridge" << std::endl;
                new_bridge = new ParallelizedBridge<DataType_SFFloat, AvePoolingBridge>(bottom_info.layer, new_layer, &layer_param,
                       &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1,
                       bottom_info.num_partitions_CPU, bottom_info.GPU_batch_sizes, bottom_info.gpu_to_device_id_map, bottom_info.data_cubes_higher, bottom_info.grad_cubes_higher,
                       share_input_output_layer);
            } else {
                std::cout << "Error in layer " << layer_name << ": currently only MAX and AVE pooling supported" << std::endl;
                assert(false);
            }
                       
            new_bridge->name = layer_name;
            if (before_first_weight_layer.count(bottom_name)) {
                before_first_weight_layer.insert(layer_name);
            }
            bridges.push_back(new_bridge);
            
            // -------------- Scheduler Update ----------------
            bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
            if (bridge_shares_data_with_prev_bridge) {
              if (bottom_info.pbridge) {
                bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
              }
            }
            bridge_name_to_info[layer_name].layer = new_layer;
            bridge_name_to_info[layer_name].pbridge = new_bridge;
            bridge_name_to_info[layer_name].GPU_batch_sizes = new_bridge->get_GPU_batch_sizes();
            bridge_name_to_info[layer_name].num_partitions_CPU = new_bridge->get_num_partitions_CPU();
            bridge_name_to_info[layer_name].gpu_to_device_id_map = new_bridge->get_used_gpu_to_device_id_map();
            bridge_name_to_info[layer_name].data_cubes_higher = new_bridge->get_data_cubes_higher();
            bridge_name_to_info[layer_name].grad_cubes_higher = new_bridge->get_grad_cubes_higher();
            increment_use(bottom_name, split_layer_to_use_count); // If this was a split, use another split next time
            // ----------End of Scheduler Update --------------
        }
        else if (layer_type == "RELU")
        {
            std::cout << "Constructing RELU bridge" << std::endl;

            // Get the previous bridge's information
            assert(layer_param.bottom_size() == 1);
            std::string bottom_name = layer_param.bottom(0);
            Layer_Info bottom_info = bridge_name_to_info[get_full_name(bottom_name, split_layer_to_use_count)];
            
            // input_[R,C,D] is the same as output_[R,C,D]
            size_t output_R = bottom_info.get_R();
            size_t output_C = bottom_info.get_C();
            size_t output_D = bottom_info.get_D();

            // Create the new layer
            new_data = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_layer = new LayerFloat(new_data, new_grad);

            // Create the new bridge
            new_bridge = new ParallelizedBridge<DataType_SFFloat, ReLUBridge>(bottom_info.layer, new_layer, &layer_param,
                       &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1,
                       bottom_info.num_partitions_CPU, bottom_info.GPU_batch_sizes, bottom_info.gpu_to_device_id_map, bottom_info.data_cubes_higher, bottom_info.grad_cubes_higher,
                       share_input_output_layer);
            
            new_bridge->name = layer_name;
            if (before_first_weight_layer.count(bottom_name)) {
                before_first_weight_layer.insert(layer_name);
            }
            bridges.push_back(new_bridge);
            
            // -------------- Scheduler Update ----------------
            bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
            if (bridge_shares_data_with_prev_bridge) {
              if (bottom_info.pbridge) {
                bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
              }
            }
            // If this bridge is in-place, still insert the name (map it to an empty layer) so that
            // we know it has been processed. Then update the current map entry for the top/bottom
            // bridge with the info from this bridge/layer. Recall in-place is for the GPU only,
            // i.e. the output layer always exists for the CPU for now.
            // SHADJIS TODO: Currently need to copy this if statement to every bridge which could
            // be in-place
            if (share_input_output_layer) {
                bridge_name_to_info[layer_name].layer = NULL;
                // Also, since this is in-place, assert that the input is not a split bridge, because
                // split bridges are only added after all in-place operations
                assert(bottom_name == get_full_name(bottom_name, split_layer_to_use_count));
                layer_name = bottom_name;
            }
            bridge_name_to_info[layer_name].layer = new_layer;
            bridge_name_to_info[layer_name].pbridge = new_bridge;
            bridge_name_to_info[layer_name].GPU_batch_sizes = new_bridge->get_GPU_batch_sizes();
            bridge_name_to_info[layer_name].num_partitions_CPU = new_bridge->get_num_partitions_CPU();
            bridge_name_to_info[layer_name].gpu_to_device_id_map = new_bridge->get_used_gpu_to_device_id_map();
            bridge_name_to_info[layer_name].data_cubes_higher = new_bridge->get_data_cubes_higher();
            bridge_name_to_info[layer_name].grad_cubes_higher = new_bridge->get_grad_cubes_higher();
            increment_use(bottom_name, split_layer_to_use_count); // If this was a split, use another split next time
            // ----------End of Scheduler Update --------------
        }
        else if (layer_type == "LRN")
        {
            std::cout << "Constructing LRN bridge" << std::endl;

            // Get the previous bridge's information
            assert(layer_param.bottom_size() == 1);
            std::string bottom_name = layer_param.bottom(0);
            Layer_Info bottom_info = bridge_name_to_info[get_full_name(bottom_name, split_layer_to_use_count)];
            
            // input_[R,C,D] is the same as output_[R,C,D]
            size_t output_R = bottom_info.get_R();
            size_t output_C = bottom_info.get_C();
            size_t output_D = bottom_info.get_D();

            // Create the new layer
            new_data = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_layer = new LayerFloat(new_data, new_grad);
            
            // Create the new bridge
            new_bridge = new ParallelizedBridge<DataType_SFFloat, LRNBridge>(bottom_info.layer, new_layer, &layer_param,
                       &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1,
                       bottom_info.num_partitions_CPU, bottom_info.GPU_batch_sizes, bottom_info.gpu_to_device_id_map, bottom_info.data_cubes_higher, bottom_info.grad_cubes_higher,
                       share_input_output_layer);
            
            new_bridge->name = layer_name;
            if (before_first_weight_layer.count(bottom_name)) {
                before_first_weight_layer.insert(layer_name);
            }
            bridges.push_back(new_bridge);
            
            // -------------- Scheduler Update ----------------
            bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
            if (bridge_shares_data_with_prev_bridge) {
              if (bottom_info.pbridge) {
                bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
              }
            }
            bridge_name_to_info[layer_name].layer = new_layer;
            bridge_name_to_info[layer_name].pbridge = new_bridge;
            bridge_name_to_info[layer_name].GPU_batch_sizes = new_bridge->get_GPU_batch_sizes();
            bridge_name_to_info[layer_name].num_partitions_CPU = new_bridge->get_num_partitions_CPU();
            bridge_name_to_info[layer_name].gpu_to_device_id_map = new_bridge->get_used_gpu_to_device_id_map();
            bridge_name_to_info[layer_name].data_cubes_higher = new_bridge->get_data_cubes_higher();
            bridge_name_to_info[layer_name].grad_cubes_higher = new_bridge->get_grad_cubes_higher();
            increment_use(bottom_name, split_layer_to_use_count); // If this was a split, use another split next time
            // ----------End of Scheduler Update --------------
        }
        else if (layer_type == "DROPOUT")
        {
            std::cout << "Constructing DROPOUT bridge" << std::endl;

            // Get the previous bridge's information
            assert(layer_param.bottom_size() == 1);
            std::string bottom_name = layer_param.bottom(0);
            Layer_Info bottom_info = bridge_name_to_info[get_full_name(bottom_name, split_layer_to_use_count)];
            
            // input_[R,C,D] is the same as output_[R,C,D]
            size_t output_R = bottom_info.get_R();
            size_t output_C = bottom_info.get_C();
            size_t output_D = bottom_info.get_D();

            // Create the new layer
            new_data = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_layer = new LayerFloat(new_data, new_grad);
            
            // Create the new bridge
            // SHADJIS TODO: I made the max threads 4 below because it was faster than 1 or 16.
            // For these smaller bridges it is usually slower to use all the threads, but need to measure.
            // Then can do something similar for ReLU, etc.
            new_bridge = new ParallelizedBridge<DataType_SFFloat, DropoutBridge>(bottom_info.layer, new_layer, &layer_param,
                       // &solver_param, driver, min<size_t>(1, corpus.mini_batch_size), 1,
                       &solver_param, driver, min<size_t>(std::min(int(hw_concurrency), 4), corpus.mini_batch_size), 1,
                       bottom_info.num_partitions_CPU, bottom_info.GPU_batch_sizes, bottom_info.gpu_to_device_id_map, bottom_info.data_cubes_higher, bottom_info.grad_cubes_higher,
                       share_input_output_layer);
            
            new_bridge->name = layer_name;
            if (before_first_weight_layer.count(bottom_name)) {
                before_first_weight_layer.insert(layer_name);
            }
            bridges.push_back(new_bridge);
            
            // -------------- Scheduler Update ----------------
            bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
            if (bridge_shares_data_with_prev_bridge) {
              if (bottom_info.pbridge) {
                bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
              }
            }
            // SHADJIS TODO: See comment in DropoutBridge_impl.hxx, currently we hard-coded this
            // assumption into the GPU dropout layer. That was not done for ReLU GPU, which works
            // even if a different top and bottom are used. Dropout also shouldn't assume that the
            // layers are shared.
            if (share_input_output_layer) {
                bridge_name_to_info[layer_name].layer = NULL;
                // Also, since this is in-place, assert that the input is not a split bridge, because
                // split bridges are only added after all in-place operations
                assert(bottom_name == get_full_name(bottom_name, split_layer_to_use_count));
                layer_name = bottom_name;
            }
            bridge_name_to_info[layer_name].layer = new_layer;
            bridge_name_to_info[layer_name].pbridge = new_bridge;
            bridge_name_to_info[layer_name].GPU_batch_sizes = new_bridge->get_GPU_batch_sizes();
            bridge_name_to_info[layer_name].num_partitions_CPU = new_bridge->get_num_partitions_CPU();
            bridge_name_to_info[layer_name].gpu_to_device_id_map = new_bridge->get_used_gpu_to_device_id_map();
            bridge_name_to_info[layer_name].data_cubes_higher = new_bridge->get_data_cubes_higher();
            bridge_name_to_info[layer_name].grad_cubes_higher = new_bridge->get_grad_cubes_higher();
            increment_use(bottom_name, split_layer_to_use_count); // If this was a split, use another split next time
            // ----------End of Scheduler Update --------------
        }
        
        // SHADJIS TODO: Currently the rest of these are not pbridges, can do that later
        
        else if (layer_type == "ELTWISE")
        {
            std::cout << "Constructing ELTWISE bridge" << std::endl;
            
            // Iterate over bottoms and ensure height/width/depth same for all
            size_t input_R = 0;
            size_t input_C = 0;
            size_t input_D = 0;
            LayerVec eltwise_input_layers;
            BridgeVector eltwise_input_pbridges;
            for (int i_bottom = 0; i_bottom < layer_param.bottom_size(); ++i_bottom) {
                Layer_Info input_layer = bridge_name_to_info[get_full_name(layer_param.bottom(i_bottom), split_layer_to_use_count)];
                increment_use(layer_param.bottom(i_bottom), split_layer_to_use_count); // If this was a split, use another split next time
                if (input_R == 0 && input_C == 0 && input_D == 0) {
                    input_R = input_layer.get_R();
                    input_C = input_layer.get_C();
                    input_D = input_layer.get_D();
                } else {
                    assert(input_R == input_layer.get_R());
                    assert(input_C == input_layer.get_C());
                    assert(input_D == input_layer.get_D());
                }
                eltwise_input_layers.push_back(input_layer.layer);
                eltwise_input_pbridges.push_back(input_layer.pbridge);
            }
            assert(input_R > 0 && input_C > 0 && input_D > 0);
            assert(eltwise_input_layers.size());
            assert(eltwise_input_pbridges.size());
            
            // Create the output layer of the concat
            new_data = new LogicalCubeFloat(input_R, input_C, input_D, B);
            new_grad = new LogicalCubeFloat(input_R, input_C, input_D, B);
            new_layer = new LayerFloat(new_data, new_grad);
            
            // Note: Eltwise never uses the input layer, i.e. we pass in an arbitrary one (e.g. 0) but this p_input_layer is never used,
            // since instead we use p_input_layers which contains all of them
            new_bridge = new EltwiseBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>(eltwise_input_layers[0],
                       new_layer, &layer_param, &solver_param, driver);
            for (size_t i = 0; i < eltwise_input_layers.size(); i++) {
              ((EltwiseBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>*)new_bridge)->p_input_layers.push_back(eltwise_input_layers[i]);
              
              // Also in the backwards pass of eltwise we copy the gradients, so connect the pointers here
              //
              // Note: we can't just set the g cube of this layer and then free the old cube because this layer may belong to a pbridge
              // A pbridge makes sub-bridges whose internal g cubes point to the original one
              // So if the input is a pbridge, need to do a "deep" set of the cube
              // We don't have this problem for concat because the depths/batches are not contiguous so we need to allocate new output cubes for fw + bw
              // And we don't have this problem for split because there the cube elimination is just a no-op, i.e. we just skip creating the output split layers at all
              // But for eltwise, we have to undo the creation of (i.e. delete) each input layer's output g cube
              //
              // There is however 1 case where there the input bridge is a pbridge (i.e. needs a deep set) but its output layer does not match the input layer:
              // to the eltwise, which is when you have a split, i.e. pbrdge -> split, and then one split goes directly to another eltwise (this happens in resnet).
              // In that case, in the info map we still give the split layer the pbridge's info, so potentially the pbridghe before the split could
              // skip copies (gpu->cpu copy @ end of its fw (for output d), and cpu->gpu copy @ start of its bw (for output g)).
              // But that means split has its own output layer, and while split's output d cube matches the pbrdge's output d cube, the split output g cube does not.
              // So the split and the pbridge into the split have different output layers. That is why the "if" below checks not only for a pbridge corresponding to 
              // the source layer, but also that the source layer is the ouput layer of that pbridge. If not (which only happens for a split), then ignore the pbridge deep 
              // set, and just set the output g cube of the split to the output g cube of the eltwise. Also note we cannot double-set, i.e. set the split's output layer
              // and the pbridge's output layer, since split bw right now first inits the input g cube to 0 and then sums all the gradients, i.e. it is not in-place.
              // Finally, note that if the input to the eltwise is a split but before the split it was a normal bridge, not a pbridge, then there is no problem, since 
              // pbridge would be NULL below so it would just set the split output anyway.
              //
              // In summary, this "if" check is equivalent to just checking if(eltwise_input_pbridges[i]) only, since that always implies 
              // eltwise_input_layers[i] == eltwise_input_pbridges[i]->p_output_layer, except for 1 case, which is when the input to the eltwise
              // is coming from a split (SHADJIS TODO: assert that), in which case the output layer of the pbridge is not the output layer of the split,
              if (eltwise_input_pbridges[i] && eltwise_input_layers[i] == eltwise_input_pbridges[i]->p_output_layer) {
                eltwise_input_pbridges[i]->update_p_output_layer_gradient_CPU_ONLY(new_layer->p_gradient_cube->get_p_data());   // deep set
              } else {
                eltwise_input_layers[i]->p_gradient_cube->set_p_data(new_layer->p_gradient_cube->get_p_data()); // will free the cube and release ownership
              }
            }
            new_bridge->name = "ELTWISE_" + layer_name;    // This name is arbitrary and does not matter
            bridges.push_back(new_bridge);

            // -------------- Scheduler Update ----------------
            // SHADJIS TODO: share data on GPU
            //bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
            //if (bridge_shares_data_with_prev_bridge) {
            //  if (bottom_info.pbridge) {
            //    bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
            //  }
            //}
            bridge_name_to_info[layer_name].layer = new_layer;
            //bridge_name_to_info[layer_name].pbridge = new_bridge;
            //bridge_name_to_info[layer_name].GPU_batch_sizes = new_bridge->get_GPU_batch_sizes();
            //bridge_name_to_info[layer_name].num_partitions_CPU = new_bridge->get_num_partitions_CPU();
            //bridge_name_to_info[layer_name].gpu_to_device_id_map = new_bridge->get_used_gpu_to_device_id_map();
            //bridge_name_to_info[layer_name].data_cubes_higher = new_bridge->get_data_cubes_higher();
            //bridge_name_to_info[layer_name].grad_cubes_higher = new_bridge->get_grad_cubes_higher();
            // ----------End of Scheduler Update --------------
        }
        else if (layer_type == "CONCAT")
        {
            std::cout << "Constructing CONCAT bridge" << std::endl;
            
            // Iterate over bottoms and ensure height/width same for all
            size_t input_R = 0;
            size_t input_C = 0;
            size_t total_D = 0;
            LayerVec concat_input_layers;
            for (int i_bottom = 0; i_bottom < layer_param.bottom_size(); ++i_bottom) {
                Layer_Info input_layer = bridge_name_to_info[get_full_name(layer_param.bottom(i_bottom), split_layer_to_use_count)];
                increment_use(layer_param.bottom(i_bottom), split_layer_to_use_count); // If this was a split, use another split next time
                if (input_R == 0 && input_C == 0) {
                    input_R = input_layer.get_R();
                    input_C = input_layer.get_C();
                } else {
                    assert(input_R == input_layer.get_R());
                    assert(input_C == input_layer.get_C());
                }
                total_D += input_layer.get_D();
                concat_input_layers.push_back(input_layer.layer);
            }
            assert(input_R > 0 && input_C > 0);
            assert(concat_input_layers.size());
            
            // Create the output layer of the concat
            new_data = new LogicalCubeFloat(input_R, input_C, total_D, B);
            new_grad = new LogicalCubeFloat(input_R, input_C, total_D, B);
            new_layer = new LayerFloat(new_data, new_grad);
            
            // Note: Concat never uses the input layer, i.e. we pass in an arbitrary one (e.g. 0) but this p_input_layer is never used,
            // since instead we use p_input_layers which contains all of them
            new_bridge = new GeneralConcatBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>(concat_input_layers[0],
                       new_layer, &layer_param, &solver_param, driver);
            for (size_t i = 0; i < concat_input_layers.size(); i++) {
              // SHADJIS TODO: Use ConcatBridge eventually (profile speeds match then merge them)
              ((GeneralConcatBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>*)new_bridge)->p_input_layers.push_back(concat_input_layers[i]);
            }
            new_bridge->name = "CONCAT_" + layer_name;    // This name is arbitrary and does not matter
            bridges.push_back(new_bridge);

            // -------------- Scheduler Update ----------------
            // SHADJIS TODO: share data on GPU
            //bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
            //if (bridge_shares_data_with_prev_bridge) {
            //  if (bottom_info.pbridge) {
            //    bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
            //  }
            //}
            bridge_name_to_info[layer_name].layer = new_layer;
            //bridge_name_to_info[layer_name].pbridge = new_bridge;
            //bridge_name_to_info[layer_name].GPU_batch_sizes = new_bridge->get_GPU_batch_sizes();
            //bridge_name_to_info[layer_name].num_partitions_CPU = new_bridge->get_num_partitions_CPU();
            //bridge_name_to_info[layer_name].gpu_to_device_id_map = new_bridge->get_used_gpu_to_device_id_map();
            //bridge_name_to_info[layer_name].data_cubes_higher = new_bridge->get_data_cubes_higher();
            //bridge_name_to_info[layer_name].grad_cubes_higher = new_bridge->get_grad_cubes_higher();
            // ----------End of Scheduler Update --------------
        }
        else if (layer_type == "BATCHNORM")
        {
            std::cout << "Constructing BATCH NORM bridge" << std::endl;

            // Get the previous bridge's information
            assert(layer_param.bottom_size() == 1);
            std::string bottom_name = layer_param.bottom(0);
            Layer_Info bottom_info = bridge_name_to_info[get_full_name(bottom_name, split_layer_to_use_count)];
            
            // input_[R,C,D] is the same as output_[R,C,D]
            size_t output_R = bottom_info.get_R();
            size_t output_C = bottom_info.get_C();
            size_t output_D = bottom_info.get_D();

            // Create the new layer
            new_data = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_layer = new LayerFloat(new_data, new_grad);

            // Create the new bridge
            new_bridge = new ParallelizedBridge<DataType_SFFloat, BatchNormBridge>(bottom_info.layer, new_layer, &layer_param,
                       &solver_param, driver, min<size_t>(1, corpus.mini_batch_size), nphysical_cores,
                       bottom_info.num_partitions_CPU, bottom_info.GPU_batch_sizes, bottom_info.gpu_to_device_id_map, bottom_info.data_cubes_higher, bottom_info.grad_cubes_higher,
                       share_input_output_layer);
            
            new_bridge->name = layer_name;
            if (before_first_weight_layer.count(bottom_name)) {
                before_first_weight_layer.insert(layer_name);
            }
            bridges.push_back(new_bridge);
            
            // -------------- Scheduler Update ----------------
            bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
            if (bridge_shares_data_with_prev_bridge) {
              if (bottom_info.pbridge) {
                bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
              }
            }
            // If this bridge is in-place, still insert the name (map it to an empty layer) so that
            // we know it has been processed. Then update the current map entry for the top/bottom
            // bridge with the info from this bridge/layer. Recall in-place is for the GPU only,
            // i.e. the output layer always exists for the CPU for now.
            // SHADJIS TODO: Currently need to copy this if statement to every bridge which could
            // be in-place
            if (share_input_output_layer) {
                bridge_name_to_info[layer_name].layer = NULL;
                // Also, since this is in-place, assert that the input is not a split bridge, because
                // split bridges are only added after all in-place operations
                assert(bottom_name == get_full_name(bottom_name, split_layer_to_use_count));
                layer_name = bottom_name;
            }
            bridge_name_to_info[layer_name].layer = new_layer;
            bridge_name_to_info[layer_name].pbridge = new_bridge;
            bridge_name_to_info[layer_name].GPU_batch_sizes = new_bridge->get_GPU_batch_sizes();
            bridge_name_to_info[layer_name].num_partitions_CPU = new_bridge->get_num_partitions_CPU();
            bridge_name_to_info[layer_name].gpu_to_device_id_map = new_bridge->get_used_gpu_to_device_id_map();
            bridge_name_to_info[layer_name].data_cubes_higher = new_bridge->get_data_cubes_higher();
            bridge_name_to_info[layer_name].grad_cubes_higher = new_bridge->get_grad_cubes_higher();
            increment_use(bottom_name, split_layer_to_use_count); // If this was a split, use another split next time
            // ----------End of Scheduler Update --------------
        }
        else if (layer_type == "SCALE")
        {
            std::cout << "Constructing SCALE bridge" << std::endl;

            // Get the previous bridge's information
            assert(layer_param.bottom_size() == 1);
            std::string bottom_name = layer_param.bottom(0);
            Layer_Info bottom_info = bridge_name_to_info[get_full_name(bottom_name, split_layer_to_use_count)];
            
            // input_[R,C,D] is the same as output_[R,C,D]
            size_t output_R = bottom_info.get_R();
            size_t output_C = bottom_info.get_C();
            size_t output_D = bottom_info.get_D();

            // Create the new layer
            new_data = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_layer = new LayerFloat(new_data, new_grad);

            // Create the new bridge
            // This one will be parallelized using multiple partitions like conv, even though it has BLAS operations,
            // because many of the updates are serial over the images
            new_bridge = new ParallelizedBridge<DataType_SFFloat, ScaleBridge>(bottom_info.layer, new_layer, &layer_param,
                       &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1,
                       bottom_info.num_partitions_CPU, bottom_info.GPU_batch_sizes, bottom_info.gpu_to_device_id_map, bottom_info.data_cubes_higher, bottom_info.grad_cubes_higher,
                       share_input_output_layer);
            
            new_bridge->name = layer_name;
            if (before_first_weight_layer.count(bottom_name)) {
                before_first_weight_layer.insert(layer_name);
            }
            bridges.push_back(new_bridge);
            
            // -------------- Scheduler Update ----------------
            bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
            if (bridge_shares_data_with_prev_bridge) {
              if (bottom_info.pbridge) {
                bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
              }
            }
            // If this bridge is in-place, still insert the name (map it to an empty layer) so that
            // we know it has been processed. Then update the current map entry for the top/bottom
            // bridge with the info from this bridge/layer. Recall in-place is for the GPU only,
            // i.e. the output layer always exists for the CPU for now.
            // SHADJIS TODO: Currently need to copy this if statement to every bridge which could
            // be in-place
            if (share_input_output_layer) {
                bridge_name_to_info[layer_name].layer = NULL;
                // Also, since this is in-place, assert that the input is not a split bridge, because
                // split bridges are only added after all in-place operations
                assert(bottom_name == get_full_name(bottom_name, split_layer_to_use_count));
                layer_name = bottom_name;
            }


            bridge_name_to_info[layer_name].layer = new_layer;
            bridge_name_to_info[layer_name].pbridge = new_bridge;
            bridge_name_to_info[layer_name].GPU_batch_sizes = new_bridge->get_GPU_batch_sizes();
            bridge_name_to_info[layer_name].num_partitions_CPU = new_bridge->get_num_partitions_CPU();
            bridge_name_to_info[layer_name].gpu_to_device_id_map = new_bridge->get_used_gpu_to_device_id_map();
            bridge_name_to_info[layer_name].data_cubes_higher = new_bridge->get_data_cubes_higher();
            bridge_name_to_info[layer_name].grad_cubes_higher = new_bridge->get_grad_cubes_higher();
            increment_use(bottom_name, split_layer_to_use_count); // If this was a split, use another split next time
            // ----------End of Scheduler Update --------------
        }
        else if (layer_type == "SOFTMAXWITHLOSS" || layer_type == "SOFTMAX")
        {
            std::cout << "Constructing SOFTMAX bridge" << std::endl;

            // Softmax bridges have 2 bottoms, but one of them is labels
            // Get the one which is in the map
            int i_bottom = 0;
            for ( ; i_bottom < layer_param.bottom_size(); ++i_bottom) {
                if (bridge_name_to_info.count(get_full_name(layer_param.bottom(i_bottom), split_layer_to_use_count))) {
                    break;
                }
            }
            assert(i_bottom != layer_param.bottom_size()); // assert a match was found
            std::string bottom_name = layer_param.bottom(i_bottom);
            Layer_Info bottom_info = bridge_name_to_info[get_full_name(bottom_name, split_layer_to_use_count)];
            
            // SHADJIS TODO: Is this layer used for anything? sizes might be arbitraty
            size_t output_R = bottom_info.get_R();
            size_t output_C = bottom_info.get_C();
            size_t output_D = bottom_info.get_D();
            new_data = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
            new_layer = new LayerFloat(new_data, new_grad);
            
            // must be initialized to point to next mini batch
            LogicalCubeFloat * const labels = new LogicalCubeFloat(NULL, 1, 1, 1, B);

            new_bridge = new SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat,
                   Layout_CRDB, CPUDriver>(bottom_info.layer, new_layer, labels, driver);
            new_bridge->name = layer_name;
            bridges.push_back(new_bridge);
            bridge_name_to_info[layer_name].layer = new_layer;
            increment_use(bottom_name, split_layer_to_use_count); // If this was a split, use another split next time
        }
        else
        {
            std::cout << "This layer type is not supported: "<< layer_type << "!" << std::endl;
            assert(false);
        }

        // If this bridge is used in multiple places it might need a split bridge
        // However if this bridge does not need back prop no need for a split bridge
        if (!before_first_weight_layer.count(layer_name)) {
            
            // Add a split bridge
            // - also caffe uses the same top in multiple bottom locations (i.e. not different top names), for example
            //   inception_3a/output in googlenet or pool1 in ResNet-50, etc.
            // - In caffe, a layer which is in-place has the same top and bottom, so before counting the number of uses
            //   and making the split bridge, make sure we have applied all the in-place operations
            // - Unlike the split bridge for FC model parallelism, this one can benefit from not copying data back to the 
            //   host: if the data is on a GPU, multiple GPUs, GPU/CPU, etc., there is no need to go through the host.
            //
            //   SHADJIS TODO: Ideally, at this split we could still share pointers even if the bridge 
            //   before the split was on the GPU (or many GPUs). I.e. before the split we might have all 
            //   of our data on GPUs 1-4, and the upcoming group will use the same batch parallelism per 
            //   bridge in the group, so we can share input data pointers (i.e. each bridge in the group 
            //   can share its input data cube with the output data cube of the prev bridge's output layer).
            //   However, we cannot share the input gradient cube of the group's bridges with the output
            //   gradient cube of the previous bridge since we need to sum up all of the gradients going
            //   backwards in the split. For this reason I will just make this a CPU bridge, although we
            //   could save a copy in the FW pass.
            //   Note: this is not true of a split before model parallelism since then we need all the
            //   data to go to each GPU, which it never would have previously been if we did data 
            //   parallelism (and if we did model parallelism before, then we must merge anyway in order
            //   to have the full data and calculate the correct gradient).
            
            // Get the name of the top, which is the name of the layer to be split
            assert(layer_param.top_size() == 1);
            std::string input_name_to_split = layer_param.top(0);
            // Assert this is not a split of a split -- if that were the case, it should have
            // instead been 1 larger split
            assert(input_name_to_split == get_full_name(input_name_to_split, split_layer_to_use_count));
            
            // Get the number of uses of this top, excluding in-place bridges (make sure we've applied them all first)
            int num_uses = 0;
            if (should_insert_split_bridge(bridge_name_to_info, input_name_to_split, net_param, num_uses)) {
            
                std::cout << "Inserting SPLIT bridge with " << num_uses << " outputs" << std::endl;

                // The reason we add a split bridge rather than just pass the layer pointer to each downstream bridge
                // is that the bw pass of a split is a sum, so we have to put the data gradient cubes somewhere. So
                // we need to make a new layer for each split, even though the data will be the same (the grads will not).
            
                // Get the information of the input to the split
                Layer_Info bottom_info = bridge_name_to_info[input_name_to_split];
                
                // input_[R,C,D] is the same as output_[R,C,D]
                size_t output_R = bottom_info.get_R();
                size_t output_C = bottom_info.get_C();
                size_t output_D = bottom_info.get_D();
                
                // Create the new layer
                // Note: Similarly to how concat never uses the input layer, split never uses the output layer
                // Therefore we can just make empty cubes
                new_data = new LogicalCubeFloat(NULL, output_R, output_C, output_D, B);
                new_grad = new LogicalCubeFloat(NULL, output_R, output_C, output_D, B);
                new_layer = new LayerFloat(new_data, new_grad);
                
                LayerFloat * input_layer_of_split_bridge = bottom_info.layer;
                new_bridge = new SplitBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>(input_layer_of_split_bridge,
                    new_layer, &layer_param, &solver_param, driver);
                    
                // Now create the output layer for each split. Recall that above the single output layer of the entire split bridge is unused,
                // (empty cubes). This is because it rather has a separate layer for each split.
                // In the fw pass, we just need to pass the data cube from the split bridge's input layer to each output data cube,
                // i.e. just make sure that the input data cube is the same (same pointer) as all the output cubes.
                // In the bw pass, we need to read the gradients from each of the output layers' gradient cubes, sum them,
                // and write that to the input gradient cube. This means we now need to allocate space for gradient but not data
                // cubes for each output layer of the split bridge
                for (int i = 0; i < num_uses; ++i) {
                  // Do not allocate
                  new_data = new LogicalCubeFloat(input_layer_of_split_bridge->p_data_cube->get_p_data(), output_R, output_C, output_D, B);
                  // Allocate
                  new_grad = new LogicalCubeFloat(output_R, output_C, output_D, B);
                  new_layer = new LayerFloat(new_data, new_grad);
                  ((SplitBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB, CPUDriver>*)new_bridge)->p_output_layers.push_back(new_layer);
                  
                  // Also make a new entry into bridge_name_to_info for this split, so we can find it later
                  // This copies information, which is redundant unfortunately
                  std::string new_split_layer_name = get_split_name(input_name_to_split, i);
                  // -------------- Scheduler Update ----------------
                  // SHADJIS TODO: can we share these?
                  //bool bridge_shares_data_with_prev_bridge = new_bridge->get_share_pointer_with_prev_bridge();
                  //if (bridge_shares_data_with_prev_bridge) {
                  //  if (bottom_info.pbridge) {
                  //    bottom_info.pbridge->set_share_pointer_with_next_bridge(true);
                  //  }
                  //}
                  bridge_name_to_info[new_split_layer_name].layer = new_layer;
                  bridge_name_to_info[new_split_layer_name].pbridge = NULL;//bottom_info.pbridge;
                  //bridge_name_to_info[new_split_layer_name].GPU_batch_sizes = bottom_info.GPU_batch_sizes;
                  //bridge_name_to_info[new_split_layer_name].num_partitions_CPU = bottom_info.num_partitions_CPU;
                  //bridge_name_to_info[new_split_layer_name].gpu_to_device_id_map = bottom_info.gpu_to_device_id_map;
                  //bridge_name_to_info[new_split_layer_name].data_cubes_higher = bottom_info.data_cubes_higher;
                  //bridge_name_to_info[new_split_layer_name].grad_cubes_higher = bottom_info.grad_cubes_higher;
                  // ----------End of Scheduler Update --------------
                }
                new_bridge->name = "SPLIT_" + input_name_to_split;    // This name is arbitrary and does not matter
                bridges.push_back(new_bridge);
                
                // We also want to make sure that we can find these new split layers later,
                // so add them to the split layers map
                split_layer_to_use_count[input_name_to_split] = 0;
            }
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
      Timer t_total_minus_first_10;

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

      size_t current_epoch = 0;    
      // std::cout << "EPOCH: " << current_epoch << std::endl;
      float loss = 0.;
      float accuracy = 0.;
    
      // Run for max_iter iterations
      for (size_t batch = 0; batch < num_batch_iterations; ++batch) {

        if (batch == 10) {
          t_total_minus_first_10.restart();
        }
      
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
      std::cout << "Total Time (minus first 10 iterations): " << t_total_minus_first_10.elapsed() << std::endl;
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
          // Print out solver as well so we know what this network is using to train
          std::cout << std::endl << "Solver:" << std::endl;
          std::string line;
          std::ifstream solver_file(file);
          while (getline(solver_file,line)) {
            if (line[0] != '#') {
              std::cout << line << std::endl;
            }
          }
          solver_file.close();
          std::cout << std::endl;

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

#endif
