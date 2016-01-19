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
#include <glog/logging.h>

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
    LogicalCube<DataType_SFFloat, Layout_CRDB> * images;
    // 1 x 1 x 1 x n_images
    LogicalCube<DataType_SFFloat, Layout_CRDB> * labels;
    // n_rows x n_cols x dim x 1
    LogicalCube<DataType_SFFloat, Layout_CRDB> * mean;

    std::string filename;

    Corpus(const cnn::LayerParameter & layer_param, const string data_binary):
      scale(layer_param.transform_param().scale()), mirror(layer_param.transform_param().mirror()),
      crop_size(layer_param.transform_param().crop_size()), phase(layer_param.include(0).phase() == 0) {
      mdb_env_source = layer_param.data_param().source(); 
      mini_batch_size = layer_param.data_param().batch_size();
      initialize_input_data_and_labels(layer_param, data_binary);
      images = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, mini_batch_size);  
    }

    ~Corpus() {

      delete images;
      delete labels;
      delete mean;
    }

    /**
     * Reset all env and cursor state of the LMBD reader
     * returns 0 on success
     */
    int OpenLmdbReader(){
      cnn::Datum datum;

      CHECK_EQ(mdb_env_create(&mdb_env_),MDB_SUCCESS) << "Error in mdb_env_create";
      CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS) << "Error in mdb_env_set_mapsize";
      CHECK_EQ(mdb_env_open(mdb_env_, mdb_env_source.c_str(), MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "Error in mdb_env_open for " << mdb_env_source;
      CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS) << "Transaction could not be started";
      CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS) << "Error in mdb_open";
      CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS) << "Error in mdb_cursor_open";
      mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);

      datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
      dim = datum.channels();
      if (crop_size != 0) {
        n_rows = crop_size;
        n_cols = crop_size;
      } else {
        n_rows = datum.height();
        n_cols = datum.width();
      }

      tmpimg = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, 1);

      // daniter TODO : return real success  here
      return 0;
    }

    void CloseLmdbReader(){
      delete tmpimg;
      mdb_txn_abort(mdb_txn_);
      mdb_cursor_close(mdb_cursor_);
    }

    /**
     * Reads the next mini batch of data from lmdb. Updates cursors accordingly.
     * Returns number of items loaded into images
     */
    int LoadLmdbData(){
      cnn::Datum datum;
      // Note that the corpus owns the storage of its images
      // daniter TODO : Is this necessary?  I think it's defaulted to this anyway (MDB_FIRST)
      MDB_cursor_op op = MDB_NEXT;;
      float * const labels_data = labels->get_p_data();
      for (size_t b = 0; b < mini_batch_size; b++) { 
          mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
          datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
          labels_data[b] = datum.label(); 
          // Process image reads the image from datum, does some preprocessing
          // and copies it into the images buffer
          process_image(images->physical_get_RCDslice(b), datum);
      }

      // daniter TODO : fix so it can return less than all images
      return images->n_elements; 
    }

  private:
    const float scale;
    const bool mirror;
    const int crop_size;
    const bool phase;
    MDB_env* mdb_env_ = NULL;
    MDB_dbi mdb_dbi_;
    MDB_txn* mdb_txn_;
    MDB_cursor* mdb_cursor_;
    MDB_val mdb_key_, mdb_value_;
    std::string mdb_env_source;
    LogicalCube<DataType_SFFloat, Layout_CRDB> * tmpimg;


    void initialize_input_data_and_labels(const cnn::LayerParameter & layer_param, const string data_binary) {
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
      num_mini_batches = ceil(float(n_images) / mini_batch_size);
      last_batch_size = mini_batch_size - (num_mini_batches * mini_batch_size - n_images);

      // Define and initialize cube storing the mean image from the database
      mean = new LogicalCube<DataType_SFFloat, Layout_CRDB>(datum.height(), datum.width(), dim, 1);
      float * const mean_data = mean->get_p_data();

      // Check for mean_file
      if (layer_param.transform_param().has_mean_file()) {
        const string & mean_file = layer_param.transform_param().mean_file();
        Parser::read_proto_from_binary_file(mean_file.c_str(), &cube);
        const int count_ = datum.height()* datum.width()* dim;
        for (int i = 0; i < count_; ++i) {
          mean_data[i] = cube.data(i);
        }
      }
      // If no mean_file, then check for individual channel means
      // SHADJIS TODO: Currently I am copying these mean values to 
      // a cube, but would be faster/use less memory to just store
      // the values for each channel and use those directly.
      else if (layer_param.transform_param().mean_value_size() > 0) {
        // Iterate over each channel and set mean value
        for (unsigned int d = 0; d < dim; ++d) {
          float channel_mean_val = 0.;
          if (d < size_t(layer_param.transform_param().mean_value_size())) {
            channel_mean_val = layer_param.transform_param().mean_value(d);
          }
          for (int px = 0; px < datum.height()*datum.width(); ++px) {
            mean_data[px + d*datum.height()*datum.width()] = channel_mean_val;
          }
        }
      }
      else {
        mean->reset_cube();
      }

      // Initialize the cube storing the correct labels
      labels = new LogicalCube<DataType_SFFloat, Layout_CRDB>(1, 1, 1, n_images);

      filename = data_binary;

      // SHADJIS TODO: Why do we use fopen/fread some places? Why not C++? Need to make consistent.

      //if (filename != "NA"){
      
      // First check if the file already exists
      FILE * pFile = fopen(filename.c_str(), "r");
      // SHADJIS TODO: If using default names, maybe do not overwrite binary since otherwise could end up 
      // using old binary by accident? E.g. if the user switches the dataset and does not specify a binary,
      // we don't want to be using an old binary, so could overwrite anyway if using default names.
      // There may be better ways to prevent that, e.g. a new option?
      //if(pFile && filename != "val_preprocessed.bin" && filename != "test_preprocessed.bin" && filename != "train_preprocessed.bin") {
      if(pFile) {
          fclose(pFile);
          
          // Warn
          if(filename == "val_preprocessed.bin" || filename == "test_preprocessed.bin" || filename == "train_preprocessed.bin") {
            std::cout << "\n** WARNING **  Data binary " << filename << " (the default name) already exists,";
            std::cout << "\n               and will not be overwritten. If the dataset for this net is different";
            std::cout << "\n               from the last time you ran, specify a new binary name (\"-b\" or \"-v\" option)";
            std::cout << "\n               or move the current " << filename << " to a new location.\n\n";
          } else {
            std::cout << "Data binary " << filename << " already exists, no need to write\n"; // Skip preprocessing
          }
          
          images = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, mini_batch_size);  // Ce: only one mini-batch in memory
          // We avoid having to write everything back to disk but we still have to read in labels
          MDB_cursor_op op = MDB_FIRST;
          float * const labels_data = labels->get_p_data();
          for (size_t b = 0; b < n_images; b++) {
            mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
            datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
            int img_label = datum.label();
            labels_data[b] = img_label;
            op = MDB_NEXT;
          }
      }
      // SHADJIS TODO: Here we preprocess the entire dataset and save in binary format. Note for Imagenet, this can take
      // hours (but only needs to be done once). The alternative is to preprocess the data in real-time rather than in advance.
      else {
          std::cout << "Data binary " << filename << " does not exist, creating\n";
          pFile = fopen(filename.c_str(), "wb+");
          if (pFile == NULL) {
              // perror("Error");
              throw std::runtime_error("File open error: " + filename + " " + strerror(errno)); // TODO: REAL MESSAGE
          }
          LogicalCube<DataType_SFFloat, Layout_CRDB> * tmpimg = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, 1);
          // Note that the corpus owns the storage of its images
          images = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, mini_batch_size);  // Ce: only one mini-batch in memory
          std::cout << "Start writing " << n_images << " to " << filename.c_str() << "..." << std::endl;
          MDB_cursor_op op = MDB_FIRST;
          float * const labels_data = labels->get_p_data();
          for (size_t b = 0; b < n_images; b++) {
              mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
              datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
              int img_label = datum.label();
              labels_data[b] = img_label;
              float * const single_input_batch = tmpimg->physical_get_RCDslice(0); // Ce: only one batch
              process_image(single_input_batch, datum);
              size_t written_bytes = fwrite(tmpimg->get_p_data(), sizeof(DataType_SFFloat), tmpimg->n_elements, pFile);
              if (written_bytes != tmpimg->n_elements) {
                perror("\nError writing data binary file");
                assert(false);
              }
              op = MDB_NEXT;
          }
          std::cout << "Finished writing images to " << filename.c_str() << "..." << std::endl;
          delete tmpimg;
          fclose(pFile);
      }
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

    // daniter TODO: Consider replacing nested loops below with a single loop (or some kind of vectorization?)
    // measure it to see if its faster
    void process_image(float * const &single_input_batch, cnn::Datum datum) {

      const string& data = datum.data();
      const int height = datum.height();
      const int width = datum.width();

      if (crop_size > 0) {
        int h_off, w_off;
        if (phase) {         // Training Phase
          h_off = rand() % (height - crop_size);// Take random patch
          w_off = rand() % (width - crop_size);
        } else {
          h_off = (height - crop_size) / 2;     // Take center
          w_off = (width - crop_size) / 2;
        }
        if (mirror && rand() % 2) {             // Mirror with probability 50%
          // Copy mirrored version
          const float * const mean_data = mean->get_p_data();
          for (size_t c = 0; c < dim; ++c) {
            for (int h = 0; h < crop_size; ++h) {
              for (int w = 0; w < crop_size; ++w) {
                int data_index = (c * height + h + h_off) * width + w + w_off;
                int top_index = (c * crop_size + h) * crop_size + (crop_size - 1 - w);

                float datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
                single_input_batch[top_index] = (datum_element - mean_data[data_index])*scale;
              }
            }
          }
        } else {
          // Normal copy
          const float * const mean_data = mean->get_p_data();
          for (size_t c = 0; c < dim; ++c) {
            for (int h = 0; h < crop_size; ++h) {
              for (int w = 0; w < crop_size; ++w) {
                int top_index = (c * crop_size + h) * crop_size + w;
                int data_index = (c * height + h + h_off) * width + w + w_off;

                float datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
                single_input_batch[top_index] = (datum_element - mean_data[data_index])*scale;
              }
            }
          }
        }
      } else { // No crop (and no mirror)
        const float * const mean_data = mean->get_p_data();
        for (size_t d = 0; d < dim; ++d) {
          for (size_t r = 0; r < n_rows; ++r) {
            for (size_t c = 0; c < n_cols; ++c) {
              const size_t data_index = d * n_rows * n_cols + r * n_cols + c;
              float datum_element = static_cast<float>(static_cast<uint8_t>(data[data_index]));
              single_input_batch[data_index] = (datum_element - mean_data[data_index])*scale;
            }
          }
        }
      }
    }
};

// #include "corpus.hxx"

#endif
