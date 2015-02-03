//
//  corpus.cpp
//  moka
//
//  Created by Firas Abuzaid on 1/29/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "corpus.h"

Corpus::Corpus(cnn::LayerParameter& layer_param) {
  initialize_input_data_and_labels(layer_param);
}

void Corpus::initialize_input_data_and_labels(cnn::LayerParameter& layer_param) {
  cnn::Datum datum;
  cnn::Cube cube;
  MDB_env* mdb_env_ = NULL;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  MDB_stat stat;

  switch (layer_param.data_param().backend()) {
    case 1:
      mdb_env_create(&mdb_env_);
      mdb_env_set_mapsize(mdb_env_, 1099511627776);
      mdb_env_open(mdb_env_, layer_param.data_param().source().c_str(), MDB_RDONLY|MDB_NOTLS, 0664);
      mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_);
      mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_);
      mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_);
      mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
      break;
    default:
      break;
  }

  switch (layer_param.data_param().backend()) {
    case 1:
      datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
      break;
    default:
      break;
  }

  dim = datum.channels();
  n_rows = datum.height();
  n_cols = datum.width();
  mini_batch_size = layer_param.data_param().batch_size();

  mdb_env_stat (mdb_env_, &stat);
  n_images = stat.ms_entries;
  num_mini_batches = ceil(float(n_images) / mini_batch_size);
  last_batch_size = mini_batch_size - (num_mini_batches * mini_batch_size - n_images);

  images = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, n_images);
  labels = new LogicalCube<DataType_SFFloat, Layout_CRDB>(1, 1, 1, n_images);
  mean = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, 1);
  
  if (layer_param.transform_param().has_mean_file()){
    const string& mean_file = layer_param.transform_param().mean_file();
    Parser::ReadProtoFromBinaryFile(mean_file.c_str(), &cube);
    const int count_ = n_rows* n_cols* dim;
    for (int i = 0; i < count_; ++i) {
      mean->p_data[i] = cube.data(i);
    }
  }
  else{
    mean->reset_cube();
  }
  

  
  MDB_cursor_op op = MDB_FIRST;

  for (size_t b = 0; b < n_images; b++) {
    mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    const string& data = datum.data();
    int img_label = datum.label();
    labels->p_data[b] = img_label;
    float * const single_input_batch = images->physical_get_RCDslice(b);
    for (size_t d = 0; d < dim; ++d) {
      for (size_t r = 0; r < n_rows; ++r) {
        for (size_t c = 0; c < n_cols; ++c) {
          //float datum_element = static_cast<float>(static_cast<uint8_t>(data[d*n_rows*n_cols+r*n_cols+c]));
          const int data_index = d*n_rows*n_cols+r*n_cols+c;
          float datum_element = static_cast<float>(static_cast<uint8_t>(data[d*n_rows*n_cols+r*n_cols+c]));
          single_input_batch[data_index] = (datum_element - mean->p_data[data_index])*0.00390625;
          //single_input_batch[d*n_rows*n_cols+r*n_cols+c] = rand()%10;
        }
      }
    }
    op = MDB_NEXT;
  }

}

Corpus::~Corpus() {
  delete images;
  delete labels;
  delete mean;
}
