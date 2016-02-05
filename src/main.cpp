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
    std::cout << "Usage:\n";
    std::cout << "./caffe-ct <train/test> <solver.prototxt> [-b train_data_binary_file] [-v validation_data_binary_file] [-i input_model_binary_file] [-o output_model_binary_file] [-t]" << "\n\n";
    std::cout << "Example:" << "\n";
    std::cout << "./caffe-ct train path/to/solver.prototxt -b data/train_data.bin -v data/val_data.bin -i models/previous_model -o models/new_model -t" << "\n\n";
    std::cout << "Option      Required?    Description" << "\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------------" << "\n";
    std::cout << "train/test  yes ........ Run the network in train mode (forward + backward) or test mode (forward only)" << "\n";
    std::cout << "solver      yes ........ Path to solver prototxt file" << "\n";
    std::cout << "-i <file>   no  ........ Like -b but specifies an input model. If not specified a new model is used." << "\n";
    std::cout << "                          -> This parameter can also be specified by --input-model=path/to/file" << "\n";
    std::cout << "-o <file>   no  ........ Like -o but specifies the output model. If not specified a trained_model.bin is created." << "\n";
    std::cout << "                          -> This parameter can also be specified by --output-model=path/to/file" << "\n";
    std::cout << "-t          no  ........ Print elapsed time per iteration." << "\n";
    std::cout << "                           -> This parameter can also be specified by --time" << "\n\n";
    exit(1);
  }

  string input_model_file;
  string output_model_file;
  bool time_iterations = false;

  boost::program_options::options_description desc("CaffeConTroll Options");
  desc.add_options()
    // Option 'input-model' and 'i' are equivalent.
    ("input-model,i", boost::program_options::value<string>(& input_model_file)->default_value("NA"),
     "Model binary (input)")
    // Option 'output-model' and 'o' are equivalent.
    ("output-model,o", boost::program_options::value<string>(& output_model_file)->default_value("NA"),
     "Model binary (output)")
    // Run with timing information
    ("time,t", boost::program_options::value<bool>(& time_iterations)->implicit_value(true),
     "Time iterations")
    ;

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (string(argv[1]) == "train") {
    DeepNet::load_and_train_network(argv[2], input_model_file, output_model_file, time_iterations);
  } else if (string(argv[1]) == "test") {
    DeepNet::load_and_test_network(argv[2], input_model_file, time_iterations);
  }

  return 0;
}
