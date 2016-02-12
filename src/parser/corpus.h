//
//  corpus.h
//
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
#include <thread>

#include "parser.h"
#include "lmdb.h"
#include "cnn.pb.h"
#include "../LogicalCube.h"

class Corpus {
  public:
    size_t n_images = 0;
    size_t n_rows = 0;
    size_t n_cols = 0;
    size_t dim = 0;
    size_t mini_batch_size = 0;

    // n_rows x n_cols x dim x n_images
    LogicalCube<DataType_SFFloat, Layout_CRDB> * images;
    // 1 x 1 x 1 x n_images
    LogicalCube<DataType_SFFloat, Layout_CRDB> * labels;
    // n_rows x n_cols x dim x 1
    LogicalCube<DataType_SFFloat, Layout_CRDB> * mean;

    std::string filename;

    Corpus(const cnn::LayerParameter & layer_param):
      scale(layer_param.transform_param().scale()), mirror(layer_param.transform_param().mirror()),
      crop_size(layer_param.transform_param().crop_size()), phase(layer_param.include(0).phase()) {
      mdb_env_source = layer_param.data_param().source(); 
      mini_batch_size = layer_param.data_param().batch_size();
      initialize_input_data_and_labels(layer_param);
      assert(n_rows != 0);
      assert(n_cols != 0);
      assert(dim != 0);
      assert(mini_batch_size != 0);
      assert(n_images != 0);
      images = new LogicalCube<DataType_SFFloat, Layout_CRDB>(n_rows, n_cols, dim, mini_batch_size); 
      // Initialize the cube storing the correct labels
      labels = new LogicalCube<DataType_SFFloat, Layout_CRDB>(1, 1, 1, mini_batch_size); 
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
      MDB_val mdb_key_, mdb_value_;   
      cnn::Datum datum;

      CHECK_EQ(mdb_env_create(&mdb_env_),MDB_SUCCESS) << "Error in mdb_env_create";
      CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS) << "Error in mdb_env_set_mapsize";
      CHECK_EQ(mdb_env_open(mdb_env_, mdb_env_source.c_str(), MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "Error in mdb_env_open for " << mdb_env_source;
      CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS) << "Transaction could not be started";
      CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS) << "Error in mdb_open";
      CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS) << "Error in mdb_cursor_open";
      int ret = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
      if (ret != 0){
        std::cerr << "Critical error trying to open the LMDB reader." << std::endl;
        return ret;
      }

      datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
      dim = datum.channels();
      if (crop_size != 0) {
        n_rows = crop_size;
        n_cols = crop_size;
      } else {
        n_rows = datum.height();
        n_cols = datum.width();
      }

      ResetCursor();
      return 0;
    }

    void CloseLmdbReader(){
      mdb_txn_abort(mdb_txn_);
      mdb_cursor_close(mdb_cursor_);
    }

    /**
     * Reads the next mini batch of data from lmdb. Updates cursors accordingly.
     * Returns number of images loaded
     * The offset is used to start filling in the images from an arbitrary spot.
     * It still fills the rest of images until it has filled up the minibatch.
     * This is used when wrapping over the data and starting at the beginning again
     */
    int LoadLmdbData(int offset = 0){
      MDB_val mdb_key_, mdb_value_;
      std::vector<std::thread>threads;
      cnn::Datum datum;
      int mdb_ret;

      // Note that the corpus owns the storage of its images
      float * const labels_data = labels->get_p_data();
      size_t count = 0;
      for (size_t b = offset; b < mini_batch_size; b++) { 
          mdb_ret = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
          if(mdb_ret != 0){
            break;
          }

          threads.emplace_back([this, mdb_value_, b, labels_data](){
            cnn::Datum datum;
            datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);

            labels_data[b] = datum.label(); 

            // Process image reads the image from datum, does some preprocessing
            // and copies it into the images buffer
            process_image(images->physical_get_RCDslice(b), datum);
          });

          op = MDB_NEXT;
          ++count;
      }

      // timing tests
      for(auto &i: threads){
	i.join();	
      } 
      // timing test end

      return count;
    }

    /*
     * The next load will start reading from the first element
     */
    void ResetCursor(){
      op = MDB_FIRST;
    }

  private:
    const float scale;
    const bool mirror;
    const int crop_size;
    const int phase;
    MDB_env* mdb_env_ = NULL;
    MDB_dbi mdb_dbi_;
    MDB_txn* mdb_txn_;
    MDB_cursor* mdb_cursor_;
    std::string mdb_env_source;
    MDB_cursor_op op = MDB_FIRST;

    void initialize_input_data_and_labels(const cnn::LayerParameter & layer_param) {
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
      // num_mini_batches = ceil(float(n_images) / mini_batch_size);
      // last_batch_size = mini_batch_size - (num_mini_batches * mini_batch_size - n_images);

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
    }

    // daniter TODO: Consider replacing nested loops below with a single loop (or some kind of vectorization?)
    // measure it to see if its faster
    void process_image(float * const &single_input_batch, cnn::Datum datum) {
      const int height = datum.height();
      const int width = datum.width();

      if (crop_size > 0) {
        int h_off, w_off;
        if (phase == 0) {         // Training Phase
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

                float datum_element;
                if (datum.data().size() != 0) {
                  datum_element = static_cast<float>(static_cast<uint8_t>(datum.data()[data_index]));
                } else {
                  datum_element = datum.float_data(data_index);
                }
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
                float datum_element;
                if (datum.data().size() != 0) {
                  datum_element = static_cast<float>(static_cast<uint8_t>(datum.data()[data_index]));
                } else {
                  datum_element = datum.float_data(data_index);
                }
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
              float datum_element;
              if (datum.data().size() != 0) {
                datum_element = static_cast<float>(static_cast<uint8_t>(datum.data()[data_index]));
              } else {
                datum_element = datum.float_data(data_index);
              }
              single_input_batch[data_index] = (datum_element - mean_data[data_index])*scale;
            }
          }
        }
      }
    }
};

// #include "corpus.hxx"

#endif
