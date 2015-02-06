#include <string>

#include "cnn.pb.h"

using namespace std;
using google::protobuf::Message;

class Parser {
  public:
    static bool read_proto_from_text_file(const char * solver_file, Message * proto);
    static void read_net_params_from_text_file(const string & param_file, Message * param);
    static bool read_proto_from_binary_file(const char * filename, Message * proto);
    static void data_setup(cnn::LayerParameter & layer_param, cnn::Datum & datum);
};
