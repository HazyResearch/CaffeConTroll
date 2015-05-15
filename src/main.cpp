//
//  main.cpp
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "DeepNet.h"

int main(int argc, const char * argv[]) {

  //srand(0); // TODO: for determinsitic

  if (argc < 3) {
    cout << "Usage: ./caffe-ct <train/test> <solver.prototxt> [-data-binary|-b data_binary_file] [-input-model|-i input_model_binary_file [-output-model|-o output_model_binary_file]" << endl;
    exit(1);
  }

  string data_binary;
  string input_model_file;
  string output_model_file;

  boost::program_options::options_description desc("CaffeConTroll Options");
  desc.add_options()
    // Option 'data-binary' and 'b' are equivalent.
    // SHADJIS TODO: Currently we preprocess the entire dataset inside
    // corpus initialize_input_data_and_labels and save it all as a preprocessed.bin binary. Maybe this
    // can be done incrementally (e.g. only preprocess the current mini-batch) to avoid long (but one-time
    // only) preprocessing for a large dataset the first time it is used.
    ("data-binary,b", boost::program_options::value<string>(& data_binary)->default_value(string(argv[1]) + "_preprocessed.bin"),
     "Processed data binary")
    // Option 'input-model' and 'im' are equivalent.
    ("input-model,i", boost::program_options::value<string>(& input_model_file)->default_value("NA"),
     "Model binary (input)")
    // Option 'output-model' and 'om' are equivalent.
    ("output-model,o", boost::program_options::value<string>(& output_model_file)->default_value("NA"),
     "Model binary (output)")
    ;

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (string(argv[1]) == "train") {
    DeepNet::load_and_train_network(argv[2], data_binary, input_model_file, output_model_file);
  } else if (string(argv[1]) == "test") {
    DeepNet::load_and_test_network(argv[2], data_binary, input_model_file);
  }

  return 0;
}
