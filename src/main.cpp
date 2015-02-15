//
//  main.cpp
//  moka
//
//  Created by Ce Zhang on 1/11/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "DeepNet.h"

int main(int argc, const char * argv[]) {
  if (argc < 3) {
    cout << "Usage: ./deepnet <train/test> <solver.prototxt>" << endl;
    exit(1);
  }

  string data_binary;
  string model_file;

  boost::program_options::options_description desc("Options for my program");
  desc.add_options()
      // Option 'data-binary' and 'b' are equivalent.
      ("data-binary,b", boost::program_options::value<string>(& data_binary)->default_value("toprocess.bin"),
          "Processed data binary")
      // Option 'model' and 'm' are equivalent.
      ("model,m", boost::program_options::value<string>(& model_file)->default_value("NA"),
          "Model binary")
      ;

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
  boost::program_options::notify(vm);

  if (string(argv[1]) == "train") {
    load_and_train_network(argv[2], data_binary, model_file);
  } else if (string(argv[1]) == "test") {
    load_and_test_network(argv[2], data_binary, model_file);
  }

  return 0;
}
