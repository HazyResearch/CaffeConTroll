#ifndef _moka_DeepNet_h
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
    static Corpus * read_corpus_from_lmdb(const cnn::NetParameter & net_param, const string data_binary, bool train) {
      if (train) {
        const cnn::LayerParameter layer_param = net_param.layer(0); // SHADJIS TODO: Should we be hard-coding layer 0 = train?
        string layer_type = layer_param.type();
        std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::toupper);
        if (layer_type == "DATA") {
          if (layer_param.include(0).phase() == 0) { // training phase
            return new Corpus(layer_param, data_binary);
          }
        }
      } else {
        const cnn::LayerParameter layer_param = net_param.layer(1);
        string layer_type = layer_param.type();
        std::transform(layer_type.begin(), layer_type.end(), layer_type.begin(), ::toupper);
        if (layer_type == "DATA") {
          if (layer_param.include(0).phase() == 1) { // testing phase
            return new Corpus(layer_param, data_binary);
          }
        }
      }
      std::cout << "No data layer present in prototxt file!" << std::endl;
      assert(false);
      return NULL;
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
               * This is for syntax compatability with Caffe about grouping.
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

              std::cout << "Constructing CONV layer with Grouping = " << grouping <<
                " (# Input Grouping=" << n_previous_groups << ")" << std::endl;

              output_R = compute_conv_next_layer_dimension(input_R, K, padding, stride),
                       output_C = compute_conv_next_layer_dimension(input_C, K, padding, stride),
                       output_D = layer_param.convolution_param().num_output();
              if (output_D % grouping != 0) {
                std::cout << "ERROR: Currently we only support the input depth \% grouping == 0." << std::endl;
                assert(false);
              }
              output_D /= grouping;

              if (grouping == n_previous_groups) {
                // SHADJIS TODO: Resize these vectors to the right size, filling them with empty vectors
                // This is needed so when I pass in e.g. prev_grad_cubes_higher_per_group[3],
                // if there were not that many groups previously it will not fail, but just
                // pass in an empty vector instead. I think this is not needed (i.e. alternatively
                // I could assert that the size is this, for all cases but the first conv layer).
                prev_data_cubes_higher_per_group.resize(n_previous_groups);
                prev_grad_cubes_higher_per_group.resize(n_previous_groups);
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
                if (grouping != 1 && n_previous_groups == 1) {
                  // in this case, we fork the single input group into multile output groups
                  prev_data_cubes_higher_per_group.resize(grouping);
                  prev_grad_cubes_higher_per_group.resize(grouping);
                  for (size_t i = 0; i < grouping; i++) {
                    // for each group, create bridges
                    next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                    next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                    next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

                    bridge = new ParallelizedBridge<DataType_SFFloat, ConvolutionBridge>
                             (prev_layers[0], next_layer, &layer_param, &solver_param, driver, min<size_t>(hw_concurrency, corpus.mini_batch_size), 1, // TODO: need a CMD line option here -- but currently we do not have the interface to do that.
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
              if (n_previous_groups != 1) {
                // if the previous group of this fully-connected layer contains multiple
                // groups, then it's the time to unify them! To do this, we introduce a
                // bridge whose only role is a funnel
                std::cout << "Constructing FUNNEL layer with grouping 1 (# Input Grouping=" << n_previous_groups << ")" << std::endl;
                output_R = input_R; output_C = input_C; output_D = input_D * n_previous_groups;
                next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
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
              }

              std::cout << "Constructing FC layer " << "(# Input Grouping=" << 1 << ")" << std::endl;

              // The R and C dimensions for a fully connected layer are always 1 x 1
              output_R = output_C = 1;
              output_D = layer_param.inner_product_param().num_output();
              next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
              next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
              next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

              prev_data_cubes_higher_per_group.resize(1);
              prev_grad_cubes_higher_per_group.resize(1);
              bridge = new ParallelizedBridge<DataType_SFFloat, FullyConnectedBridge>
                       (prev_layers[0], next_layer, &layer_param, &solver_param, driver, min<size_t>(1, corpus.mini_batch_size), hw_concurrency,
                       prev_num_partitions_CPU, prev_GPU_batch_sizes, prev_gpu_to_device_id_map, prev_data_cubes_higher_per_group[0], prev_grad_cubes_higher_per_group[0],
                       share_input_output_layer);

              //bridge = new FullyConnectedBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>(prev_layers[0],
              //  next_layer, &layer_param, &solver_param);

              bridge->name = layer_param.name();
              bridge->run_with_n_threads = hw_concurrency;  // TODO: Add a better abstraction here.
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
              prev_data_cubes_higher_per_group.push_back(pbridges_from_this_iteration[0]->get_data_cubes_higher());
              prev_grad_cubes_higher_per_group.push_back(pbridges_from_this_iteration[0]->get_grad_cubes_higher());
              pbridges_from_this_iteration.clear();
              // ----------End of Scheduler Update --------------
              before_first_weight_layer = false;
          }
          else if (layer_type == "POOLING")
          {
              std::cout << "Constructing MAXPOOLING " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

              const size_t K = layer_param.pooling_param().kernel_size(), stride = layer_param.pooling_param().stride();

              output_R = compute_conv_next_layer_dimension(input_R, K, 0, stride),
                       output_C = compute_conv_next_layer_dimension(input_C, K, 0, stride);

              prev_data_cubes_higher_per_group.resize(n_previous_groups);
              prev_grad_cubes_higher_per_group.resize(n_previous_groups);
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

              prev_data_cubes_higher_per_group.resize(n_previous_groups);
              prev_grad_cubes_higher_per_group.resize(n_previous_groups);
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

              prev_data_cubes_higher_per_group.resize(n_previous_groups);
              prev_grad_cubes_higher_per_group.resize(n_previous_groups);
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

              prev_data_cubes_higher_per_group.resize(n_previous_groups);
              prev_grad_cubes_higher_per_group.resize(n_previous_groups);
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
    static void train_network(const BridgeVector & bridges, const Corpus & corpus, const cnn::NetParameter & net_param,
        const cnn::SolverParameter & solver_param, const string input_model_file, const string snapshot_file_name,
        const Corpus & val_corpus, bool time_iterations = false) {

      SoftmaxBridge * const softmax = (SoftmaxBridge *) bridges.back();
      Bridge * const first = (Bridge *) bridges.front();

      LogicalCubeFloat * const labels = softmax->p_data_labels;
      LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;
      
      // Also make a temporary labels array for when we wrap around the training set
      // SHADJIS TODO: Can do this in other ways, can time to see if slow.
      float *labels_buffer = new float [corpus.mini_batch_size];

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
      
      // Open the file for the first time during training
      FILE * pFile = fopen (corpus.filename.c_str(), "rb");
      if (!pFile)
        throw std::runtime_error("Error opening the corpus file: " + corpus.filename);

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

        // Read in the next mini-batch from file
        size_t rs = fread(corpus.images->get_p_data(), sizeof(DataType_SFFloat), corpus.images->n_elements, pFile);
        // initialize labels for this mini batch
        labels->set_p_data(corpus.labels->physical_get_RCDslice(current_image_location_in_dataset));
        // If we read less than we expected, read the rest from the beginning
        size_t num_floats_left_to_read = corpus.images->n_elements - rs;
        if (num_floats_left_to_read > 0) {
            // Increment epoch
            ++current_epoch;
            // Close the file and re-open it
            fclose(pFile);
            pFile = fopen (corpus.filename.c_str(), "rb");
            if (!pFile)
              throw std::runtime_error("Error opening the corpus file: " + corpus.filename);
            // Read the remaining data from the file, adjusting the pointer to where we
            // read until previously as well as the amount to read
            size_t rs2 = fread((float *) (corpus.images->get_p_data()) + rs, sizeof(DataType_SFFloat), num_floats_left_to_read, pFile);
            assert(rs2 == num_floats_left_to_read);
            // Also, we need to copy over the labels to a contiguous memory location
            // The labels are all allocated in corpus.labels. Normally we just set
            // the data pointer of our local "labels" cube to the right place in the
            // corpus labels cube data. But since the labels we want aren't anywhere
            // contiguously, we can allocate an array for them.
            // First, copy the correct labels to the array
            // This involves 2 steps: the end portion of the training set and the 
            // beginning portion.
            
            // Check if we actually read nothing (i.e. we were right at the end before)
            // In this case, we don't have to copy anything else
            if (rs == 0) {
                assert(current_image_location_in_dataset == 0);
                memcpy(labels_buffer, corpus.labels->physical_get_RCDslice(0), sizeof(float) * corpus.mini_batch_size);
            }
            // Otherwise, we have to copy twice
            else {
                size_t num_images_from_end = corpus.n_images - current_image_location_in_dataset;
                assert(num_images_from_end > 0);
                assert(num_images_from_end < corpus.mini_batch_size);
                size_t num_images_from_beginning = corpus.mini_batch_size - num_images_from_end;
                memcpy(labels_buffer,
                    corpus.labels->physical_get_RCDslice(current_image_location_in_dataset),
                    sizeof(float) * num_images_from_end);
                memcpy(labels_buffer + num_images_from_end,
                    corpus.labels->physical_get_RCDslice(0),
                    sizeof(float) * num_images_from_beginning);
            }
            // Now point labels to this array
            labels->set_p_data(labels_buffer);
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
        input_data->set_p_data(mini_batch);

        softmax->reset_loss();

        // forward pass
        for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
          // (*bridge)->set_curr_batch_size(curr_batch_size);
          (*bridge)->forward();
#ifdef _LAYER_PROFILING
         (*bridge)->report_forward();
#endif
        }

        t_forward = t.elapsed();

        loss += (softmax->get_loss() / float(corpus.mini_batch_size));
        accuracy += float(DeepNet::find_accuracy(labels, (*--bridges.end())->p_output_layer->p_data_cube)) / float(corpus.mini_batch_size);

        // backward pass
        t.restart();
        for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
          // (*bridge)->set_curr_batch_size(curr_batch_size);
          (*bridge)->backward();
#ifdef _LAYER_PROFILING
          (*bridge)->report_backward();
#endif
        }
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
      
      fclose(pFile);
      delete labels_buffer;
      std::cout << "Total Time (seconds): " << t_total.elapsed() << std::endl;
    }

      static Corpus * load_network(const char * file, const string & data_binary, cnn::SolverParameter & solver_param,
          cnn::NetParameter & net_param, BridgeVector & bridges, bool train) {

        // not necessary if being called from load_and_(train|test)_network,
        // but necessary for certain tests
        DeepNetConfig::train_ = train;

        if (Parser::read_proto_from_text_file(file, &solver_param) &&
            Parser::read_net_params_from_text_file(solver_param.net(), &net_param)) {
          Corpus * corpus = DeepNet::read_corpus_from_lmdb(net_param, data_binary, train);

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


      static float test_network(const BridgeVector & bridges, const Corpus & corpus, const cnn::NetParameter & net_param,
          const cnn::SolverParameter & solver_param, bool time_iterations = false) {

        // TODO: we need a more general AbstractLossBridge
        SoftmaxBridge * const softmax = (SoftmaxBridge *) bridges.back();
        Bridge * const first = (Bridge *) bridges.front();

        LogicalCubeFloat * const labels = softmax->p_data_labels;
        LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;

        FILE * pFile = fopen(corpus.filename.c_str(), "rb");
        if (!pFile)
          throw std::runtime_error("Error opening the corpus file: " + corpus.filename);
        
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

          size_t num_elements_read = fread(corpus.images->get_p_data(), sizeof(DataType_SFFloat), corpus.images->n_elements, pFile);
          assert(num_elements_read == corpus.images->n_elements);
          //t_load = t.elapsed();
          //t.restart();
          float * const mini_batch = corpus.images->physical_get_RCDslice(0);
          input_data->set_p_data(mini_batch);

          softmax->reset_loss();

          // initialize labels for this mini batch
          labels->set_p_data(corpus.labels->physical_get_RCDslice(corpus_batch_index));
          // forward pass
          for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
            (*bridge)->forward();
          }
          //t_forward = t.elapsed();

          float loss = (softmax->get_loss() / corpus.mini_batch_size);
          total_loss += loss;
          int batch_accuracy = DeepNet::find_accuracy(labels, softmax->p_output_layer->p_data_cube);
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
        fclose(pFile);
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
      static void load_and_train_network(const char * file, const string data_binary, const string input_model_file,
            const string output_model_file, const string val_data_binary, bool time_iterations = false) {
        DeepNetConfig::train_ = true;

        BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
        Corpus * const corpus = DeepNet::load_network(file, data_binary, solver_param, net_param, bridges, true);

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
            val_corpus = DeepNet::read_corpus_from_lmdb(net_param, val_data_binary, false);
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

      static float load_and_test_network(const char * file, const string data_binary, const string input_model_file, bool time_iterations = false) {
        DeepNetConfig::train_ = false;

        BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
        Corpus * const corpus = DeepNet::load_network(file, data_binary, solver_param, net_param, bridges, false);

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
