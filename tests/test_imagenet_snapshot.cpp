#include "../src/DeepNet.h"
#include "gtest/gtest.h"
#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/DropoutBridge.h"
#include "test_types.h"
#include "snapshot_parser.h"


TEST(ImageNetSnapshotTest, RunTest) {

  BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
  // char const * file = "tests/imagenet_train/imagenet_solver_2.prototxt";
  char const * file = "tests/imagenet_train/imagenet_snapshot_solver.prototxt";
  const std::string data_binary = "imagenet.bin";

  Corpus * corpus = load_network(file, data_binary, solver_param, net_param, bridges, true);

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

    FILE * pFile = fopen (corpus->filename.c_str(), "rb");
    if(!pFile)
      throw runtime_error("Error opening the corpus file: " + corpus->filename);

    // num_mini_batches - 1, because we need one more iteration for the final mini batch
    // (the last mini batch may not be the same size as the rest of the mini batches)
    for (size_t batch = 0, corpus_batch_index = 0; batch < corpus->num_mini_batches ; ++batch,
        corpus_batch_index += corpus->mini_batch_size) {
      Timer t;
      Timer t2;

      // The last batch may be smaller, but all other batches should be the appropriate size.
      // rs will then contain the real number of entires
      size_t rs = fread(corpus->images->p_data, sizeof(DataType_SFFloat), corpus->images->n_elements, pFile);
      if (rs != corpus->images->n_elements && batch != corpus->num_mini_batches - 1){
        std::cout << "Error in reading data from " << corpus->filename << " in batch " << batch << " of " << corpus->num_mini_batches << std::endl;
        std::cout << "read:  " << rs << " expected " << corpus->images->n_elements << std::endl;
        exit(1);
      }

      t_load = t.elapsed();

      t.restart();
      // initialize input_data for this mini batch
      // Ce: Notice the change here compared with the master branch -- this needs to be refactored
      // to make the switching between this and the master branch (that load everything in memory)
      // dynamically and improve code reuse.
      float * const mini_batch = corpus->images->physical_get_RCDslice(0);
      input_data->p_data = mini_batch;

      softmax->reset_loss();

      // initialize labels for this mini batch
      labels->p_data = corpus->labels->physical_get_RCDslice(corpus_batch_index);

      // forward pass
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {

        //std::cout << (*bridge)->name << "    " << (*bridge)->p_output_layer->p_data_cube->n_elements << std::endl;
        (*bridge)->forward();

        //(*bridge)->report_forward();
        //(*bridge)->report_forward_last_transfer.print();
        //if((*bridge)->name == "relu1"){
        //  for(int i=0;i< (*bridge)->p_output_layer->p_data_cube->n_elements; i++){
        //    std::cout << i << "    " << (*bridge)->p_output_layer->p_data_cube->p_data[i] << std::endl;
        //  }
        //}
      }

      t_forward = t.elapsed();

      float loss = (softmax->get_loss() / corpus->mini_batch_size);
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
                  << "    " << 1.0*accuracy/corpus->mini_batch_size;
        std::cout << "\033[0m" << std::endl;
      }

    }

    fclose(pFile);

    // TODO: handle the very last batch, which may not have the same
    // batch size as the rest of the batches
    cout << "Average Time (seconds) per Epoch: " << t_total.elapsed()/(epoch+1) << endl;
  }
  cout << "Total Time (seconds): " << t_total.elapsed() << endl;

  // double output;
  // if (expected_accuracy.is_open()) {
  //   expected_accuracy >> output;
  //   EXPECT_NEAR(acc, output, 0.05);
  // }else{
  //   FAIL();
  // }

}