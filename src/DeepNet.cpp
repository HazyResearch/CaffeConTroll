#include "DeepNet.h"

using namespace std;

// computes the output dimension for any convolution layer
inline size_t compute_conv_next_layer_dimension(const size_t R_i, const size_t K,
    const size_t padding, const size_t stride ) {
  return (R_i + 2 * padding - K) / stride + 1;
}

// load training data into Corpus object, return Corpus object
// Note: we assume that the very first layer in the .protoxt
// file specifies the data layer
// TODO: also read in test set
Corpus * read_corpus_from_lmdb(const cnn::NetParameter & net_param, const string data_binary, bool train) {
  if (train){
    const cnn::LayerParameter layer_param = net_param.layers(0);
    if (layer_param.type() == cnn::LayerParameter_LayerType_DATA) {
      if (layer_param.include(0).phase() == 0) { // training phase
        return new Corpus(layer_param, data_binary);
      }
    }  
  }
  else{
    const cnn::LayerParameter layer_param = net_param.layers(1);
    if (layer_param.type() == cnn::LayerParameter_LayerType_DATA) {
      if (layer_param.include(0).phase() == 1) { // testing phase
        return new Corpus(layer_param, data_binary);
      }
    }  
  }
  cout << "No data layer present in prototxt file!" << endl;
  assert(false);
  return NULL;
}

//// Shubham: Need to be refactored a bit on the basis of how these features would actually be used.
/// Should we have a separate test function?
void write_model_to_file(const BridgeVector bridges, const string model_file){
  FILE * pFile;
  pFile = fopen (model_file.c_str(), "wb");
  LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
  LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
  for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
    model = (*bridge)->get_model_cube();
    if(model){
      fwrite (model->p_data , sizeof(DataType_SFFloat), model->n_elements, pFile);  
    }
    bias = (*bridge)->get_bias_cube();
    if(bias){
      fwrite (bias->p_data , sizeof(DataType_SFFloat), bias->n_elements, pFile); 
    }
  }
  fclose(pFile);
}

void read_model_from_file(BridgeVector & bridges, const string model_file){
  FILE * pFile;
  pFile = fopen (model_file.c_str(), "rb");
  LogicalCube<DataType_SFFloat, Layout_CRDB> * model;
  LogicalCube<DataType_SFFloat, Layout_CRDB> * bias;
  for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
    model = (*bridge)->get_model_cube();
    if(model){
      fread(model->p_data , sizeof(DataType_SFFloat), model->n_elements, pFile);  
    }
    bias = (*bridge)->get_bias_cube();
    if(bias){
      fread(bias->p_data , sizeof(DataType_SFFloat), bias->n_elements, pFile); 
    }
  }
  fclose(pFile);
}

int find_accuracy(const LogicalCubeFloat * const labels, const LogicalCubeFloat * output) {
  const float* actual_data = output->p_data;
  const float* expected_label = labels->p_data;
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
      if (data_vector[k].second == static_cast<int>(expected_label[i])) {
        ++accuracy;
        break;
      }
    }
  }
  return accuracy;
  //cout << "Accuracy: " << (accuracy / num) << endl;
}

// This takes in the bridge vector (which has been initialized to be empty in load_and_train_network)
// and builds up a list of bridges in the vector in the order in which they will be executed in the forward
// pass. Only the bridges variable is modified.
void construct_network(BridgeVector & bridges, Corpus & corpus, const cnn::NetParameter & net_param,
  const cnn::SolverParameter & solver_param) {
  size_t input_R = corpus.n_rows, input_C = corpus.n_cols, input_D = corpus.dim, B = corpus.mini_batch_size;
          //, last_B = corpus.last_batch_size;

  // Create the Logical Cubes for the initial data layer
  LogicalCubeFloat * prev_data = new LogicalCubeFloat(corpus.images->physical_get_RCDslice(0), input_R, input_C, input_D, B);
  LogicalCubeFloat * prev_grad = new LogicalCubeFloat(input_R, input_C, input_D, B);

  std::vector<Layer<DataType_SFFloat, Layout_CRDB> *> prev_layers, next_layers;
  prev_layers.push_back(new Layer<DataType_SFFloat, Layout_CRDB>(prev_data, prev_grad));

  //Layer<DataType_SFFloat, Layout_CRDB> * prev_layer = new Layer<DataType_SFFloat, Layout_CRDB>(prev_data, prev_grad);

  const size_t num_layers = net_param.layers_size();

  AbstractBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> * bridge = NULL;
  LogicalCube<DataType_SFFloat, Layout_CRDB> * next_data = NULL;
  LogicalCube<DataType_SFFloat, Layout_CRDB> * next_grad = NULL;
  Layer<DataType_SFFloat, Layout_CRDB> * next_layer = NULL;

  size_t output_R = input_R, output_C = input_C, output_D = input_D;
  bool is_first_conv = true;

  for (size_t i_layer = 0; i_layer < num_layers; ++i_layer) {

    const cnn::LayerParameter layer_param = net_param.layers(i_layer);
    const cnn::LayerParameter_LayerType layer_type = layer_param.type();

    const size_t n_previous_groups = prev_layers.size();

    if (layer_type != cnn::LayerParameter_LayerType_DATA) {
      switch (layer_type) {
        // Note: These braces surrounding each case statement are necessary
        // because we're initializing variables (such as config) inside the case.
        // (Otherwise, the compiler will complain about a "switch case is in protected
        // scope" error.)
        {
          case cnn::LayerParameter_LayerType_CONVOLUTION:
          const size_t K = layer_param.convolution_param().kernel_size(),
                  padding = layer_param.convolution_param().pad(),
                  stride = layer_param.convolution_param().stride(),
                  grouping = layer_param.convolution_param().group();

            std::cout << "Constructing CONV layer with Grouping = " << grouping << 
              " (# Input Grouping=" << n_previous_groups << ")" << std::endl;

            output_R = compute_conv_next_layer_dimension(input_R, K, padding, stride),
            output_C = compute_conv_next_layer_dimension(input_C, K, padding, stride),
            output_D = layer_param.convolution_param().num_output();
            if(output_D % grouping != 0){
              std::cout << "ERROR: Currently we only support the input depth \% grouping == 0." << std::endl;
              assert(false);
            }
            output_D /= grouping;

            if(grouping == n_previous_groups){
              // if input group == output group, then for each 
              // input group, create a separate bridge and a 
              // seperate output bridge
              for(size_t i=0;i<n_previous_groups;i++){
                // for each group, create bridges
                next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

                bridge = new ParallelizedBridge<DataType_SFFloat,
                  ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
                  (prev_layers[i], next_layer, &layer_param, &solver_param, 16, 1); // TODO: need a CMD line option here -- but currently we do not have the interface to do that.
                bridge->name = layer_param.name();
                bridge->needs_to_calc_backward_grad = !is_first_conv; // for the first CONV layer, do not need to calc grad for backward step
                bridges.push_back(bridge);
                next_layers.push_back(next_layer);
              }
              is_first_conv = false;
            }else{
              if(grouping != 1 && n_previous_groups == 1){
                // in this case, we fork the single input group into multile output groups
                for(size_t i=0;i<grouping;i++){
                  // for each group, create bridges
                  next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                  next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
                  next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

                  bridge = new ParallelizedBridge<DataType_SFFloat,
                    ConvolutionBridge<CPU_CONV_LOWERINGTYPE1, FUNC_NOFUNC, DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
                    (prev_layers[0], next_layer, &layer_param, &solver_param, 16, 1); // TODO: need a CMD line option here -- but currently we do not have the interface to do that.
                  bridge->name = layer_param.name();
                  bridge->needs_to_calc_backward_grad = !is_first_conv; // for the first CONV layer, do not need to calc grad for backward step

                  bridges.push_back(bridge);
                  next_layers.push_back(next_layer);
                }
                is_first_conv = false;
              }else{
                std::cout << "ERROR: Currently we do not support the case where input group is " << n_previous_groups
                  << " and output group is " << grouping << " for CONV layer..." << std::endl; 
                assert(false);
              }
            }
        }
        break;
        {
          case cnn::LayerParameter_LayerType_INNER_PRODUCT:

            if(n_previous_groups != 1){
              // if the previous group of this fully-connected layer contains multiple
              // groups, then it's the time to unify them! To do this, we introduce a 
              // bridge whose only role is a funnel 
              std::cout << "Constructing FUNNEL layer with grouping 1 (# Input Grouping=" << n_previous_groups << ")" << std::endl;
              output_R = input_R; output_C = input_C; output_D = input_D * n_previous_groups;
              next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
              next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
              next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
              bridge = new FunnelBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>(prev_layers[0],
                next_layer, &layer_param, &solver_param);
              for(size_t i=0;i<n_previous_groups;i++){
                ((FunnelBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>*)bridge)->p_input_layers.push_back(prev_layers[i]);
              }
              bridge->name = "FUNNEL";
              bridges.push_back(bridge);
              input_D = output_D;
              prev_layers.clear();
              prev_layers.push_back(next_layer);
            }

            std::cout << "Constructing FC layer " << "(# Input Grouping=" << 1 << ")" << std::endl;

            // The R and C dimensions for a fully connected layer are always 1 x 1
            output_R = output_C = 1;
            output_D = layer_param.inner_product_param().num_output();
            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, output_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

            //bridge = new ParallelizedBridge<DataType_SFFloat,
            //  FullyConnectedBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
            //  (prev_layer, next_layer, &layer_param, 16, 1); // TODO: need a CMD line option here -- but currently we do not have the interface to do that.

            bridge = new FullyConnectedBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB>(prev_layers[0],
              next_layer, &layer_param, &solver_param);
            bridge->name = layer_param.name();
            bridge->run_with_n_threads = 16;  // TODO: Add a better abstraction here.
            bridges.push_back(bridge);
            next_layers.push_back(next_layer);
        }
        break;
        {
          case cnn::LayerParameter_LayerType_POOLING:

            std::cout << "Constructing MAXPOOLING " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

            const size_t K = layer_param.pooling_param().kernel_size(), stride = layer_param.pooling_param().stride();

            output_R = compute_conv_next_layer_dimension(input_R, K, 0, stride),
                     output_C = compute_conv_next_layer_dimension(input_C, K, 0, stride);

            for(size_t i=0;i<n_previous_groups;i++){
              // input_D same as output_D
              next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, input_D, B);
              next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(output_R, output_C, input_D, B);
              next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

              bridge = new ParallelizedBridge<DataType_SFFloat,
                MaxPoolingBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >(prev_layers[i],
                    next_layer, &layer_param, &solver_param, 16, 1);
              bridge->name = layer_param.name();
              bridges.push_back(bridge);
              next_layers.push_back(next_layer);
            }

        }
        break;
        {
          case cnn::LayerParameter_LayerType_RELU:
            // input_[R,C,D] is the same as output_[R,C,D]

            std::cout << "Constructing RELU layer " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

            for(size_t i=0;i<n_previous_groups;i++){

              next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
              next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
              next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);

              bridge = new ParallelizedBridge<DataType_SFFloat,
                ReLUBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
                (prev_layers[i], next_layer, &layer_param, &solver_param, 16, 1); // TODO: need a CMD line option here -- but currently we do not have the interface to do that.
              bridge->name = layer_param.name();

              bridges.push_back(bridge);
              next_layers.push_back(next_layer);
            }
            /*
            bridge = new ReLUBridge<DataType_SFFloat, Layout_CRDB,
                   DataType_SFFloat, Layout_CRDB>(prev_layer, next_layer, &layer_param);
            */
        }
        break;
        {
          case cnn::LayerParameter_LayerType_LRN:
            // input_[R,C,D] is the same as output_[R,C,D]

            std::cout << "Constructing LRN layer " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

            for(size_t i=0;i<n_previous_groups;i++){

              next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
              next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
              next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
              //bridge = new ParallelizedLRNBridge<DataType_SFFloat>(prev_layer, next_layer, &layer_param, 4, 2);
              //bridge = new LRNBridge<DataType_SFFloat, Layout_CRDB,
              //       DataType_SFFloat, Layout_CRDB>(prev_layer, next_layer, &layer_param);
              bridge = new ParallelizedBridge<DataType_SFFloat,
                LRNBridge<DataType_SFFloat, Layout_CRDB, DataType_SFFloat, Layout_CRDB> >
                (prev_layers[i], next_layer, &layer_param, &solver_param, 16, 1); // TODO: need a CMD line option here -- but currently we do not have the interface to do that.
              bridge->name = layer_param.name();

              bridges.push_back(bridge);
              next_layers.push_back(next_layer);

            }
        }
        break;
        {
          case cnn::LayerParameter_LayerType_DROPOUT:

            std::cout << "Constructing DROPOUT layer " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

            // input_[R,C,D] is the same as output_[R,C,D]
            for(size_t i=0;i<n_previous_groups;i++){

              next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
              next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
              next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
              bridge = new DropoutBridge<DataType_SFFloat, Layout_CRDB,
                   DataType_SFFloat, Layout_CRDB>(prev_layers[i], next_layer, &layer_param, &solver_param);
              bridge->name = layer_param.name();

              bridges.push_back(bridge);
              next_layers.push_back(next_layer);
            }
        }
        break;
        {
          case cnn::LayerParameter_LayerType_SOFTMAX_LOSS:

            std::cout << "Constructing SOFTMAX layer " << "(# Input Grouping=" << n_previous_groups << ")" << std::endl;

            // input_[R,C,D] is the same as output_[R,C,D]
            if(n_previous_groups != 1){
              std::cout << "ERROR: Currently, we only support FC layer to connect " <<
                "between multiple input groups to a single output group." << std::endl;
                assert(false);
            }

            next_data = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_grad = new LogicalCube<DataType_SFFloat, Layout_CRDB>(input_R, input_C, input_D, B);
            next_layer = new Layer<DataType_SFFloat, Layout_CRDB>(next_data, next_grad);
            // must be initialized to point to next mini batch
            LogicalCubeFloat * const labels = new LogicalCubeFloat(NULL, 1, 1, 1, B);

            bridge = new SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,
                 DataType_SFFloat, Layout_CRDB>(prev_layers[0], next_layer, labels);
            bridge->name = layer_param.name();

            bridges.push_back(bridge);
            next_layers.push_back(next_layer);
        }
        break;
        default:
        cout << "This layer type is not supported: "<< layer_type << "!" << endl;
        assert(false);
      }

      // Appending the bridge to our vector of bridges, and updating pointers
      // and values for the next iteration.
      //bridges.push_back(bridge);
      
      input_R = output_R, input_C = output_C, input_D = output_D;
      //prev_data = next_data, prev_grad = next_grad;
      //prev_layer = next_layer;
      //ReadModelFromFile(bridges);

      /**
       * Swap next_layers with prev_layers and empty next;
       */
      prev_layers.clear();
      for(size_t i=0;i<next_layers.size();i++){
        prev_layers.push_back(next_layers[i]);
      }
      next_layers.clear();
    }
  }
}

// Here, we train our CNN: we iterate over the vector of bridges, forwards and backward for each batch size.
// Right now, we do this in a single-thread fashion. TODO: Create a Scheduler class, that schedules workers
// for each batch size, so that we can perform these forward and backward passes in parallel.
void train_network(const BridgeVector & bridges, const Corpus & corpus, const cnn::NetParameter & net_param,
    const cnn::SolverParameter & solver_param) {

  SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const softmax =
    (SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) bridges.back();

  AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const first =
    (AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) bridges.front();

  LogicalCubeFloat * const labels = softmax->p_data_labels;
  LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;

  float t_load;
  float t_forward;
  float t_backward;
  float t_pass;

  Timer t_total;
  const int display_iter = 50;

  const size_t num_epochs = solver_param.max_iter();
  for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
    cout << "EPOCH: " << epoch << endl;

    FILE * pFile;
    pFile = fopen (corpus.filename.c_str(), "rb");

    // num_mini_batches - 1, because we need one more iteration for the final mini batch
    // (the last mini batch may not be the same size as the rest of the mini batches)
    for (size_t batch = 0, corpus_batch_index = 0; batch < corpus.num_mini_batches - 1; ++batch,
        corpus_batch_index += corpus.mini_batch_size) {
      Timer t;
      Timer t2;

      // this loading appears to take just ~ 0.1 s for each batch,
      // so double-buffering seems an overkill here because the following operations took seconds...
      size_t rs = fread(corpus.images->p_data, sizeof(DataType_SFFloat), corpus.images->n_elements, pFile);
      if (rs != corpus.images->n_elements){
        std::cout << "Error in reading data" << std::endl;
        exit(1);
      }

      t_load = t.elapsed();

      t.restart();
      // initialize input_data for this mini batch
      // Ce: Notice the change here compared with the master branch -- this needs to be refactored
      // to make the switching between this and the master branch (that load everything in memory)
      // dynamically and improve code reuse.
      float * const mini_batch = corpus.images->physical_get_RCDslice(0);
      input_data->p_data = mini_batch;

      softmax->reset_loss();

      // initialize labels for this mini batch
      labels->p_data = corpus.labels->physical_get_RCDslice(corpus_batch_index);

      // forward pass
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        
        (*bridge)->forward();
        //(*bridge)->report_forward();
        //(*bridge)->report_forward_last_transfer.print();
      }

      t_forward = t.elapsed();

      float loss = (softmax->get_loss() / corpus.mini_batch_size);
      int accuracy = find_accuracy(labels, (*--bridges.end())->p_output_layer->p_data_cube);

      // backward pass
      t.restart();
      for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
        (*bridge)->backward();
        //(*bridge)->report_backward();
      }
      t_backward = t.elapsed();
      
      t_pass = t2.elapsed();


      if(batch % display_iter == 0){
        cout << "BATCH: " << batch << endl;
        std::cout << "Loading Time (seconds)     : " << t_load << std::endl;
        std::cout << "Forward Pass Time (seconds) : " << t_forward << std::endl;
        std::cout << "Backward Pass Time (seconds): " << t_backward << std::endl;
        std::cout << "\033[1;31m";
        std::cout << "Total Time & Loss & Accuracy: " << t_pass << "    " << loss 
                  << "    " << 1.0*accuracy/corpus.mini_batch_size;
        std::cout << "\033[0m" << std::endl;
      }
      
    }

    fclose(pFile);

    // TODO: handle the very last batch, which may not have the same
    // batch size as the rest of the batches
    cout << "Average Time (seconds) per Epoch: " << t_total.elapsed()/(epoch+1) << endl;
  }
  cout << "Total Time (seconds): " << t_total.elapsed() << endl;

}

Corpus * load_network(const char * file, const string & data_binary, cnn::SolverParameter & solver_param,
    cnn::NetParameter & net_param, BridgeVector & bridges, bool train) {

  if (Parser::read_proto_from_text_file(file, &solver_param) &&
  Parser::read_net_params_from_text_file(solver_param.net(), &net_param)) {
    Corpus * corpus = read_corpus_from_lmdb(net_param, data_binary, train);

#ifdef _DO_WARNING
    cout << "Corpus train loaded" << endl;
    cout << "CORPUS NUM IMAGES: " << corpus.n_images << endl;
    cout << "CORPUS NUM ROWS: " << corpus.n_rows << endl;
    cout << "CORPUS NUM COLS: " << corpus.n_cols << endl;
    cout << "CORPUS NUM CHANNELS: " << corpus.dim << endl;
    cout << "CORPUS MINI BATCH SIZE: " << corpus.mini_batch_size << endl;
    cout << "CORPUS NUM MINI BATCHES: " << corpus.num_mini_batches << endl;
    cout << "CORPUS LAST BATCH SIZE: " << corpus.last_batch_size << endl;
#endif

    construct_network(bridges, *corpus, net_param, solver_param);

    return corpus;
  } else {
    throw runtime_error("Error parsing the solver.protoxt file or train_val.txt file");
    return NULL;
  }
}


float test_network(const BridgeVector & bridges, const Corpus & corpus, const cnn::NetParameter & net_param,
    const cnn::SolverParameter & solver_param) {

  // TODO: we need a more general AbstractLossBridge
  SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const softmax =
    (SoftmaxLossBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) bridges.back();

  AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> * const first =
    (AbstractBridge<DataType_SFFloat, Layout_CRDB,DataType_SFFloat, Layout_CRDB> *) bridges.front();

  LogicalCubeFloat * const labels = softmax->p_data_labels;
  LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;

  FILE * pFile;
  pFile = fopen(corpus.filename.c_str(), "rb");

  // num_mini_batches - 1, because we need one more iteration for the final mini batch
  // (the last mini batch may not be the same size as the rest of the mini batches)
  int total_accuracy = 0;
  for (size_t batch = 0, corpus_batch_index = 0; batch < corpus.num_mini_batches - 1; ++batch,
      corpus_batch_index += corpus.mini_batch_size) {
    cout << "BATCH: " << batch << endl;

    Timer t;
    Timer t2;

    fread(corpus.images->p_data, sizeof(DataType_SFFloat), corpus.images->n_elements, pFile);
    std::cout << "Loading Time (seconds)     : " << t.elapsed() << std::endl;

    float * const mini_batch = corpus.images->physical_get_RCDslice(0);
    input_data->p_data = mini_batch;

    softmax->reset_loss();

    // initialize labels for this mini batch
    labels->p_data = corpus.labels->physical_get_RCDslice(corpus_batch_index);
    // forward pass
    for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
      //(*bridge)->p_input_layer->p_gradient_cube->reset_cube();
      //(*bridge)->p_output_layer->p_data_cube->reset_cube();
      (*bridge)->forward();
    }
    std::cout << "Forward Pass Time (seconds) : " << t.elapsed() << std::endl;

    float loss = (softmax->get_loss() / corpus.mini_batch_size);
    int batch_accuracy = find_accuracy(labels, softmax->p_output_layer->p_data_cube);
    total_accuracy += batch_accuracy;

    std::cout << "\033[1;31m";
    std::cout << "Total Time & Loss & Accuracy: " << t2.elapsed() << "    " << loss 
              << "    " << 1.0*batch_accuracy/corpus.mini_batch_size;
    std::cout << "\033[0m" << std::endl;

  }
  float acc = (1.0*total_accuracy/((corpus.num_mini_batches - 1)*corpus.mini_batch_size));
  cout << "Overall Accuracy " << acc << endl;
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
// 3) For epoch = 0 -> num_epochs (<- extracted from prototxt file)
//      For batch = 0 -> num_batches - 1 (<- extracted from protoxt file)
//        Compute forward pass (iterate through vector of Bridge pointers)
//      Compute forward pass for last batch (might not have the same
//                                           size as the rest of batches)
//      For batch = 0 -> num_batches - 1 (<- extracted from protoxt file)
//        Compute backward pass (iterate through vector of Bridge
//                               pointers backwards)
//      Compute backward pass for last batch (again, might not have the same
//                                            size as the rest of batches)
//
void load_and_train_network(const char * file, const string data_binary, const string model_file) {

  BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
  Corpus * corpus = load_network(file, data_binary, solver_param, net_param, bridges, true);

  // Step 3:
  // Now, the bridges vector is fully populated
  train_network(bridges, *corpus, net_param, solver_param);
  if(model_file == "NA")
    write_model_to_file(bridges, "deepnetmodel.bin");  
  else
    write_model_to_file(bridges, model_file);
  // Step 4:
  // Clean up! TODO: free the allocated bridges, layers, and cubes
}

float load_and_test_network(const char * file, const string data_binary, const string model_file) {

  BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
  Corpus * corpus = load_network(file, data_binary, solver_param, net_param, bridges, false);

  if(model_file != "NA"){
    read_model_from_file(bridges, model_file); 
    return test_network(bridges, *corpus, net_param, solver_param); 
  }
  else{
    cout << "No valid model file provided" << endl;
    assert(false);
    return -1;
  }  
}
