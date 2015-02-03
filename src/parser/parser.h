#include <string>

#include "cnn.pb.h"

using namespace std;
using google::protobuf::Message;

class Parser {
  public:
    static bool ReadProtoFromTextFile(const char* solver_file, Message* proto); 
    static void ReadNetParamsFromTextFile(const string& param_file, Message* param);
    static bool ReadProtoFromBinaryFile(const char* filename, Message* proto);
    static void DataSetup(cnn::LayerParameter& layer_param, cnn::Datum& datum);
};

