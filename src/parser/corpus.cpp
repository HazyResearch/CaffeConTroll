//
//  corpus.cpp
//  moka
//
//  Created by Firas Abuzaid on 1/29/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "corpus.h"

#ifndef MDB_NOTLS
#define 	MDB_NOTLS   0x200000
#endif

void Corpus::process_image(const cnn::LayerParameter & layer_param, float * const &single_input_batch, cnn::Datum datum){
  const string& data = datum.data();
  const int crop_size = layer_param.transform_param().crop_size();
  const int height = datum.height();
  const int width = datum.width();
  const float scale = layer_param.transform_param().scale();
  const bool mirror = layer_param.transform_param().mirror();

  if (layer_param.transform_param().has_crop_size()) {
    n_rows = crop_size;
    n_cols = crop_size;
  } else {
    n_rows = datum.height();
    n_cols = datum.width();
  }

  if (crop_size > 0) {
    int h_off, w_off;
    if (layer_param.include(0).phase() == 0) {         // Training Phase
      h_off = rand() % (height - crop_size);
      w_off = rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && rand() % 2) {
      // Copy mirrored version
      for (size_t c = 0; c < dim; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = (c * crop_size + h) * crop_size + (crop_size - 1 - w);

            float datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
            single_input_batch[top_index] = (datum_element - mean->p_data[data_index])*scale;
          }
        }
      }
    } else {
    // Normal copy
      for (size_t c = 0; c < dim; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = (c * crop_size + h) * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;

            float datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
            single_input_batch[top_index] = (datum_element - mean->p_data[data_index])*scale;
          }
        }
      }
    }
  } else {
    for (size_t d = 0; d < dim; ++d) {
      for (size_t r = 0; r < n_rows; ++r) {
        for (size_t c = 0; c < n_cols; ++c) {
          const size_t data_index = d * n_rows * n_cols + r * n_cols + c;
          float datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
          single_input_batch[data_index] = (datum_element - mean->p_data[data_index])*scale;
        }
      }
    }
  }
}

Corpus::Corpus(const cnn::LayerParameter & layer_param, const string data_binary) {
  initialize_input_data_and_labels(layer_param, data_binary);
}

/**
 * This is
 */
void Corpus::initialize_input_data_and_labels(const cnn::LayerParameter & layer_param, const string data_binary) {
  cnn::Datum datum;
  cnn::Cube cube;
  MDB_env* mdb_env_ = NULL;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  MDB_stat stat;

  // TODO
  //  - This needs refactoring to make this an option
  //  - Need help from either Firas or Shubham because this requires
  //    changing the cmd input parsing part

  switch (layer_param.data_param().backend()) {
    case 1:
      CHECK_EQ(mdb_env_create(&mdb_env_),MDB_SUCCESS) << "Error in mdb_env_create";
      CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS) << "Error in mdb_env_set_mapsize";
      CHECK_EQ(mdb_env_open(mdb_env_, layer_param.data_param().source().c_str(), MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "Error in mdb_env_open for " << layer_param.data_param().source();
      CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS) << "Transaction could not be started";
      CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS) << "Error in mdb_open";
      CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS) << "Error in mdb_cursor_open";
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
  mini_batch_size = layer_param.data_param().batch_size();
  const int crop_size = layer_param.transform_param().crop_size();

  if (layer_param.transform_param().has_crop_size()) {
    n_rows = crop_size;
    n_cols = crop_size;
  } else {
    n_rows = datum.height();
    n_cols = datum.width();
  }

  mdb_env_stat(mdb_env_, &stat);
  n_images = stat.ms_entries;
  n_images = 2000;
  num_mini_batches = ceil(float(n_images) / mini_batch_size);
  last_batch_size = mini_batch_size - (num_mini_batches * mini_batch_size - n_images);

  // Define and initialize cube storing the mean image from the database
  mean = new LogicalCube<DataType_SFFloat, Layout_CRDB>(datum.height(), datum.width(), dim, 1);

  if (layer_param.transform_param().has_mean_file()) {
    const string & mean_file = layer_param.transform_param().mean_file();
    Parser::read_proto_from_binary_file(mean_file.c_str(), &cube);
    const int count_ = datum.height()* datum.width()* dim;
    for (int i = 0; i < count_; ++i) {
      mean->p_data[i] = cube.data(i);
    }
  } else {
    mean->reset_cube();
  }

  // Initialize the cube storing the correct labels
  labels = new LogicalCube<DataType_SFFloat, Layout_CRDB>(1, 1, 1, n_images);

  filename = data_binary;

  //if (filename != "NA"){
    FILE * pFile = fopen (filename.c_str(), "wb+");
    if (pFile == NULL) {
      // perror("Error");
      throw std::runtime_error("File open error: " + filename + " " + strerror(errno)); // TODO: REAL MESSAGE
    }
    LogicalCube<DataType_SFFloat, Layout_CRDB> * tmpimg = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, 1);
    images = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, mini_batch_size);  // Ce: only one batch in memory
    std::cout << "Start writing " << n_images << " to " << filename.c_str() << "..." << std::endl;
    MDB_cursor_op op = MDB_FIRST;
    for (size_t b = 0; b < n_images; b++) {
      mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
      datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
      int img_label = datum.label();
      labels->p_data[b] = img_label;
      float * const single_input_batch = tmpimg->physical_get_RCDslice(0);  // Ce: only one batch
      process_image(layer_param, single_input_batch, datum);
      fwrite (tmpimg->p_data , sizeof(DataType_SFFloat), tmpimg->n_elements, pFile);
      op = MDB_NEXT;
    }
    std::cout << "Finished writing images to " << filename.c_str() << "..." << std::endl;

    fclose(pFile);
  //}
  // else{
  //   images = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, n_images);
  //   MDB_cursor_op op = MDB_FIRST;
  //   for (size_t b = 0; b < n_images; b++) {
  //     mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
  //     datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  //     int img_label = datum.label();
  //     labels->p_data[b] = img_label;
  //     float * const single_input_batch = images->physical_get_RCDslice(b);  // Ce: only one batch
  //     process_image(layer_param, single_input_batch, datum);
  //     op = MDB_NEXT;
  //   }
  // }


}


// TODO: This function is the original one. This should be switched dymanically with
// the previous one with reasonable code reuse.
/*
void Corpus::initialize_input_data_and_labels(const cnn::LayerParameter & layer_param) {
  cnn::Datum datum;
  cnn::Cube cube;
  MDB_env* mdb_env_ = NULL;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  MDB_stat stat;

  std::cout << mdb_version(NULL, NULL, NULL) << std::endl;
  std::cout << layer_param.data_param().source().c_str() << std::endl;

  switch (layer_param.data_param().backend()) {
    case 1:
      int rs;
      rs = mdb_env_create(&mdb_env_);
      std::cout << "2" << " " << rs << std::endl;
      rs = mdb_env_set_mapsize(mdb_env_, 1099511627776);
      std::cout << "3" << " " << rs << std::endl;
      rs = mdb_env_open(mdb_env_, layer_param.data_param().source().c_str(), MDB_RDONLY|MDB_NOTLS, 777);
      std::cout << "4" << " " << rs << std::endl;
      rs = mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_);
      std::cout << "5" << " " << rs << std::endl;
      rs = mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_);
      std::cout << "6" << " " << rs << std::endl;
      rs = mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_);
      std::cout << "7" << " " << rs  << std::endl;
      mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
      std::cout << "8" << std::endl;
      break;
    default:
      break;
  }

  std::cout << "-" << std::endl;

  switch (layer_param.data_param().backend()) {
    case 1:
      datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
      break;
    default:
      break;
  }

  std::cout << "2" << std::endl;

  dim = datum.channels();
  mini_batch_size = layer_param.data_param().batch_size();
  const int crop_size = layer_param.transform_param().crop_size();
  const int height = datum.height();
  const int width = datum.width();
  const float scale = layer_param.transform_param().scale();
  const bool mirror = layer_param.transform_param().mirror();

  if (layer_param.transform_param().has_crop_size()) {
    n_rows = crop_size;
    n_cols = crop_size;
  } else {
    n_rows = datum.height();
    n_cols = datum.width();
  }

  mdb_env_stat (mdb_env_, &stat);
  n_images = stat.ms_entries;
  num_mini_batches = ceil(float(n_images) / mini_batch_size);
  last_batch_size = mini_batch_size - (num_mini_batches * mini_batch_size - n_images);

  images = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, n_images);
  labels = new LogicalCube<DataType_SFFloat, Layout_CRDB>(1, 1, 1, n_images);
  mean = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, 1);

  if (layer_param.transform_param().has_mean_file()) {
    const string & mean_file = layer_param.transform_param().mean_file();
    Parser::read_proto_from_binary_file(mean_file.c_str(), &cube);
    const int count_ = n_rows* n_cols* dim;
    for (int i = 0; i < count_; ++i) {
      mean->p_data[i] = cube.data(i);
    }
  } else {
    mean->reset_cube();
  }

  //std::cout << n_images << std::endl;

  return;

  MDB_cursor_op op = MDB_FIRST;
  std::cout << n_images << std::endl;
  for (size_t b = 0; b < n_images; b++) {
    //std::cout << mdb_key_ << std::endl;
    //std::cout << mdb_value_ << std::endl;
    mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    const string& data = datum.data();
    int img_label = datum.label();
    labels->p_data[b] = img_label;
    float * const single_input_batch = images->physical_get_RCDslice(b);
    if (crop_size > 0) {
      int h_off, w_off;
      if (layer_param.include(0).phase() == 0) {         // Training Phase
        h_off = rand() % (height - crop_size);
        w_off = rand() % (width - crop_size);
      } else {
        h_off = (height - crop_size) / 2;
        w_off = (width - crop_size) / 2;
      }
      if (mirror && rand() % 2) {
        // Copy mirrored version
        for (size_t c = 0; c < dim; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int data_index = (c * height + h + h_off) * width + w + w_off;
              int top_index = (c * crop_size + h) * crop_size + (crop_size - 1 - w);

              float datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
              single_input_batch[top_index] = (datum_element - mean->p_data[data_index])*scale;
            }
          }
        }
      } else {
      // Normal copy
        for (size_t c = 0; c < dim; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = (c * crop_size + h) * crop_size + w;
              int data_index = (c * height + h + h_off) * width + w + w_off;

              float datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
              single_input_batch[top_index] = (datum_element - mean->p_data[data_index])*scale;
            }
          }
        }
      }
    } else {
      for (size_t d = 0; d < dim; ++d) {
        for (size_t r = 0; r < n_rows; ++r) {
          for (size_t c = 0; c < n_cols; ++c) {
            const size_t data_index = d * n_rows * n_cols + r * n_cols + c;
            float datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
            single_input_batch[data_index] = (datum_element - mean->p_data[data_index])*scale;
          }
        }
      }
    }
    op = MDB_NEXT;
  }
}
*/

Corpus::~Corpus() {
  delete images;
  delete labels;
  delete mean;
}
