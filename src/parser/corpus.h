//
//  corpus.h
//  moka
//
//  Created by Firas Abuzaid on 1/29/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef _CORPUS_H
#define _CORPUS_H

#include <string>
#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <fstream>
#include <math.h>
#include <stdint.h>
#include <iostream>
#include <fcntl.h>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message_lite.h>
#include "parser.h"
#include "lmdb.h"
#include "cnn.pb.h"


#include "../LogicalCube.h"

class Corpus {
  public:

    size_t n_images;
    size_t n_rows;
    size_t n_cols;
    size_t dim;
    size_t mini_batch_size;
    size_t num_mini_batches;
    size_t last_batch_size;

    // n_rows x n_cols x dim x n_images
    LogicalCube<DataType_SFFloat, Layout_CRDB>* images;
    // 1 x 1 x 1 x n_images
    LogicalCube<DataType_SFFloat, Layout_CRDB>* labels;
    // n_rows x n_cols x dim x 1
    LogicalCube<DataType_SFFloat, Layout_CRDB>* mean;

    explicit Corpus(const cnn::LayerParameter & layer_param);
    ~Corpus();

  private:
    void initialize_input_data_and_labels(const cnn::LayerParameter & layer_param);
};

#endif
