#include "test_types.h"
#include "gtest/gtest.h"

#include "../src/DeepNet.h"
#include "../src/Kernel.h"
#include "../src/LogicalCube.h"
#include "../src/Layer.h"
#include "../src/config.h"
#include "../src/Connector.h"
#include "../src/bridges/DropoutBridge.h"
#include "../snapshot-parser/simple_parse.h"

const string snapshot_dir("data/snapshots");

void compare_to_expected(const LogicalCube<float, Layout_CRDB> * const actual,
    const blob_map & expected) {
   EXPECT_NEAR(actual->n_elements, expected.nValues, 0);
   for (int i = 0; i < expected.nValues; ++i) {
     EXPECT_NEAR(actual->p_data[i], expected.values[i], EPS);
   }
}

void copy_blob_to_cube(const LogicalCube<float, Layout_CRDB> * const cube,
    const blob_map & blob) {
   assert((int) cube->n_elements == blob.nValues);
   for (int i = 0; i < blob.nValues; ++i) {
     cube->p_data[i] = blob.values[i];
   }
}


void check_update(const string & filename, GradientUpdater<float> * const updater) {
 std::ifstream i(filename);
 if (i.fail()) { std::cout << "Failed to open file!" << filename << std::endl; exit(-1); }
 update_file f(i);
 i.close();

 const float stepsize = updater->get_stepsize();
 EXPECT_NEAR(f.get_local_rate(), stepsize, 1e-8);
 if (dynamic_cast<SGDGradientUpdater<float> *>(updater)) {
   const float momentum = ((SGDGradientUpdater<float> *) updater)->get_momentum();
   EXPECT_NEAR(f.get_momentum(), momentum, 1e-8);
 } else if (dynamic_cast<NesterovUpdater<float> *>(updater)) {
   const float momentum = ((NesterovUpdater<float> *) updater)->get_momentum();
   EXPECT_NEAR(f.get_momentum(), momentum, 1e-8);
 }
}

void check_regularization(const string & filename, GradientUpdater<float> * const updater) {
 std::ifstream i(filename);
 if (i.fail()) { std::cout << "Failed to open file!" << filename << std::endl; exit(-1); }
 regularized_update_file r(i);
 i.close();

 const float decay = updater->get_weight_decay();
 EXPECT_NEAR(r.get_local_regu(), decay, 1e-8);
}

TEST(ImageNetSnapshotTest, RunTest) {

  BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
  char const * file = "tests/imagenet_train/imagenet_snapshot_solver.prototxt";
  const std::string data_binary = "imagenet.bin";

  Corpus * corpus = DeepNet::load_network(file, data_binary, solver_param, net_param, bridges, true);

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

  const string metadata_filename = metadata_file::generate_filename(snapshot_dir);
  std::ifstream i(metadata_filename);
  if (i.fail()) { std::cout << "Failed to open file!" << metadata_filename << std::endl; exit(-1); }
  metadata_file mf(i);
  i.close();

  const int num_params = mf.get_nParams();

  const size_t num_epochs = solver_param.max_iter();
  for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
    cout << "EPOCH: " << epoch << endl;

    FILE * pFile = fopen (corpus->filename.c_str(), "rb");
    if (!pFile)
      throw runtime_error("Error opening the corpus file: " + corpus->filename);

    // num_mini_batches - 1, because we need one more iteration for the final mini batch
    // (the last mini batch may not be the same size as the rest of the mini batches)
    for (size_t batch = 0, corpus_batch_index = 0; batch < 1; ++batch,
        corpus_batch_index += corpus->mini_batch_size) {
      Timer t;
      Timer t2;

      // The last batch may be smaller, but all other batches should be the appropriate size.
      // rs will then contain the real number of entires
      size_t rs = fread(corpus->images->p_data, sizeof(DataType_SFFloat), corpus->images->n_elements, pFile);
      if (rs != corpus->images->n_elements && batch != corpus->num_mini_batches - 1) {
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
      const int iter = epoch + batch;

      // force inputs for first layer
      const string data_filename = forward_file::generate_filename(snapshot_dir, iter, "data");
      std::ifstream i(data_filename);
      if (i.fail()) { std::cout << "Failed to open file!" << data_filename << std::endl; exit(-1); }
      forward_file ff(i);
      i.close();

      Bridge * const first = (Bridge *) bridges.front();
      copy_blob_to_cube(first->p_input_layer->p_data_cube, ff.get_output()[0]);

      // forward pass
      for (auto bridge = bridges.begin(); bridge != bridges.end(); ++bridge) {
        Bridge * const curr_bridge = *bridge;

        const string forward_filename = forward_file::generate_filename(snapshot_dir, iter, curr_bridge->name);
        std::ifstream i(forward_filename);
        if (i.fail()) { std::cout << "Failed to open file!" << forward_filename << std::endl; exit(-1); }
        forward_file ff(i);
        i.close();

        cout << curr_bridge->name << " FORWARD" << endl;

        const LogicalCube<float, Layout_CRDB> * const input = curr_bridge->p_input_layer->p_data_cube;
        cout << curr_bridge->name << " FORWARD input" << endl;
        compare_to_expected(input, ff.get_input()[0]);

        if (curr_bridge->get_model_cube() != NULL) {
          const LogicalCube<float, Layout_CRDB> * const model = curr_bridge->get_model_cube();
          cout << curr_bridge->name << " FORWARD model" << endl;
          compare_to_expected(model, ff.get_model()[0]);
        }

        if (curr_bridge->get_bias_cube() != NULL) {
          const LogicalCube<float, Layout_CRDB> * const bias = curr_bridge->get_bias_cube();
          cout << curr_bridge->name << " FORWARD bias" << endl;
          compare_to_expected(bias, ff.get_model()[1]);
        }

        curr_bridge->forward();

        const LogicalCube<float, Layout_CRDB> * const output = curr_bridge->p_output_layer->p_data_cube;
        cout << curr_bridge->name << " FORWARD output" << endl;
        compare_to_expected(output, ff.get_output()[0]);

        // force inputs for next layer
        copy_blob_to_cube(output, ff.get_output()[0]);
      }

      t_forward = t.elapsed();

      float loss = (softmax->get_loss() / corpus->mini_batch_size);
      int accuracy = DeepNet::find_accuracy(labels, (*--bridges.end())->p_output_layer->p_data_cube);

      // backward pass
      t.restart();
      string prev_name = ""; int param_id = -2;
      for (auto bridge = bridges.rbegin(); bridge != bridges.rend(); ++bridge) {
        Bridge * const curr_bridge = *bridge;
        const string name = curr_bridge->name;
        const string backward_filename = backward_file::generate_filename(snapshot_dir, iter, curr_bridge->name);
        std::ifstream i(backward_filename);
        if (i.fail()) { std::cout << "Failed to open file!" << backward_filename << std::endl; exit(-1); }
        backward_file bf(i);
        i.close();

        const LogicalCube<float, Layout_CRDB> * const input_grad = curr_bridge->p_input_layer->p_gradient_cube;
        compare_to_expected(input_grad, bf.get_input_g()[0]);

        if (curr_bridge->get_model_grad_cube() != NULL) {
          const LogicalCube<float, Layout_CRDB> * const model_grad = curr_bridge->get_model_grad_cube();
          compare_to_expected(model_grad, bf.get_model_g()[0]);
        }

        if (curr_bridge->get_bias_grad_cube() != NULL) {
          const LogicalCube<float, Layout_CRDB> * const bias_grad = curr_bridge->get_bias_grad_cube();
          compare_to_expected(bias_grad, bf.get_model_g()[1]);
        }

        if (name != prev_name && curr_bridge->get_model_cube() != NULL) {
          param_id += 2; // for now we assume that there's always a bias cube
        }

        if (curr_bridge->get_model_cube() != NULL) {
          string filename = update_file::generate_filename(snapshot_dir, iter, name, num_params - param_id - 2);
	  string filename_regu = regularized_update_file::generate_filename(snapshot_dir, iter, name,
              num_params - param_id - 2);
          GradientUpdater<float> * const model_updater = curr_bridge->get_model_updater();
          check_update(filename, model_updater);
	  check_regularization(filename_regu, model_updater);
        }

        if (curr_bridge->get_bias_cube() != NULL) {
          string filename = update_file::generate_filename(snapshot_dir, iter, name, num_params - param_id - 1);
	  string filename_regu = regularized_update_file::generate_filename(snapshot_dir, iter, name,
              num_params - param_id - 1);
          GradientUpdater<float> * const bias_updater = curr_bridge->get_bias_updater();
          check_update(filename, bias_updater);
          check_regularization(filename_regu, bias_updater);
	}
        prev_name = name;
        (*bridge)->backward();
      }

      t_backward = t.elapsed();

      t_pass = t2.elapsed();

      if (batch % display_iter == 0) {
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

}
