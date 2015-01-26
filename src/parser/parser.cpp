
#include "parser.h"
#include "lmdb.h"
#include "cnn.pb.h"

#include <stdint.h>
#include <iostream>
#include <fcntl.h>
#include <fstream>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message_lite.h>

using namespace std;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool Parser::ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  google::protobuf::io::FileInputStream fileInput(fd);
  fileInput.SetCloseOnDelete( true );
  bool success = google::protobuf::TextFormat::Parse(&fileInput, proto);
  return success;
}

void Parser::ReadNetParamsFromTextFile(const string& param_file, Message* param) {
  ReadProtoFromTextFile(param_file.c_str(), param);
}

void Parser::DataSetup(cnn::LayerParameter& layer_param, cnn::Datum& datum){
  MDB_env* mdb_env_ = NULL; //TODO: resolve compiler warning about not being initialized
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;

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

