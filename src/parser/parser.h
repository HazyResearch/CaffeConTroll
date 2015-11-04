
#ifndef _PARSER_H
#define _PARSER_H


#include <glog/logging.h>
#include <stdint.h>
#include <iostream>
#include <fcntl.h>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message_lite.h>

#include "lmdb.h"
#include "cnn.pb.h"

#ifndef NDB_NOTLS
#define 	MDB_NOTLS   0x200000
#endif


using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

class Parser {
  public:
    static bool read_proto_from_text_file(const char * filename, Message * proto) {
      int fd = open(filename, O_RDONLY);
      google::protobuf::io::FileInputStream fileInput(fd);
      fileInput.SetCloseOnDelete(true);
      bool success = google::protobuf::TextFormat::Parse(&fileInput, proto);
      return success;
    }

    static bool read_net_params_from_text_file(const string & param_file, Message * param) {
      return read_proto_from_text_file(param_file.c_str(), param);
    }

    static bool read_proto_from_binary_file(const char * filename, Message * proto) {
      int fd = open(filename, O_RDONLY);
      google::protobuf::io::ZeroCopyInputStream* raw_input = new FileInputStream(fd);
      google::protobuf::io::CodedInputStream* coded_input = new CodedInputStream(raw_input);
      coded_input->SetTotalBytesLimit(1073741824, 536870912);
      bool success = proto->ParseFromCodedStream(coded_input);
      return success;
    }

    static void data_setup(cnn::LayerParameter & layer_param, cnn::Datum & datum) {
      MDB_env* mdb_env_ = NULL;
      MDB_dbi mdb_dbi_;
      MDB_txn* mdb_txn_;
      MDB_cursor* mdb_cursor_;
      MDB_val mdb_key_, mdb_value_;

      switch (layer_param.data_param().backend()) {
        case 1:
          CHECK_EQ(mdb_env_create(&mdb_env_),MDB_SUCCESS) << "Error in mdb_env_create";
          CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS) << "Error in mdb_env_set_mapsize";
          CHECK_EQ(mdb_env_open(mdb_env_, layer_param.data_param().source().c_str(), MDB_RDONLY|MDB_NOTLS, 777), MDB_SUCCESS) << "Error in mdb_env_open";
          CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS) << "Transaction could not be started";
          CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS) << "Error in mdb_open";
          CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS) << "Error in mdb_cursor_open";
          mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
          break;
        default:
          break;
      }

      // Read a data point, and use it to initialize the top blob.
      switch (layer_param.data_param().backend()) {
        case 1:
          datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
          break;
        default:
          break;
      }
      mdb_env_close(mdb_env_);
    }
};

// #include "parser.hxx"

#endif