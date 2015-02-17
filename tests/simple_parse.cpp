#include <fstream>
#include <cstdlib>
#include <string>
#include <iostream>
#include <string.h>
#include <fstream>
#include <stdio.h>
#include <cassert>
#include <stdexcept>

const std::string szStart("START");
const std::string szEnd("END");
const std::string szLoss("LOSS");
const std::string szModel("MODEL_BLOBS");
const std::string szInput("INPUT_BLOBS");
const std::string szOutput("OUTPUT_BLOBS");
const std::string szForward("FORWARD");
const std::string szBlob("BLOB");

/* These are accessors that consume datatypes from the stream */

// This checks for a statement "key"
void
getStatement(std::istream &in, const std::string key) {
  std::string this_key;
  in >> this_key;
  if(key != this_key) 
    throw std::runtime_error("[getStatement] Expected " + key + " but found " + this_key);
}

void
putStatement(std::ostream &o, const std::string key) {
  o << key << std::endl;
}



void 
getKey(std::istream &in, const std::string key, double &v) {
  std::string this_key; 
  in >> this_key >> v;
  if(key != this_key) {
    throw std::runtime_error("[getKey.d] Expected " + key + " but found " + this_key);
  }
}
void
putKey(std::ostream &o, const std::string key, const double v) {
  o << key << " " << v << std::endl;
}

void 
getKey(std::istream &in, const std::string key, int &v) {
  std::string this_key; 
  in >> this_key >> v;
  if(key != this_key) {
    throw std::runtime_error("[getKey.i] Expected " + key + " but found " + this_key);
  }
}

void
putKey(std::ostream &o, const std::string key, const int v) {
  o << key << " " << v << std::endl;
}

void
getKey(std::istream &in, const std::string key, std::string &v) {
  std::string this_key; 
  in >> this_key >> v;
  if(key != this_key) {
    throw std::runtime_error("[getKey.i] Expected " + key + " but found " + this_key);
  }
}

void
putKey(std::ostream &o, const std::string key, const std::string v) {
  o << key << " " << v << std::endl;
}


// ****************
// End simple types
// ****************

// The only compound structures are blob maps
// blob maps are an encoding of arrays.
struct blob_map {
  int* indexes;
  double *values;
  int nValues;
};

void parse_blob_map(std::istream &in, struct blob_map &bm) {
  int nValues = 0;
  getKey(in, szBlob, nValues);
  assert(nValues > 0);
  getStatement(in, szStart);
  bm.nValues = nValues;
  bm.indexes = new int[nValues];
  bm.values  = new double[nValues];
  std::cerr << "\t nValues=" << nValues << std::endl;
  for(int i = 0; i < nValues; i++) {
    int index; double value;
    in >> index >> value;
    assert(index >= 0);
    bm.indexes[i] = index;
    bm.values[i]  = value;
  }
   getStatement(in, szEnd);
}

// Printing 
void blob_map_print(std::ostream &o, const blob_map &bm) {
  putKey(o, szBlob, bm.nValues);
  putStatement(o, szStart);

  for(int i = 0; i < bm.nValues; i++) {
    o << "  " << bm.indexes[i] << " " << bm.values[i] << std::endl;
  }
  putStatement(o, szEnd);
}
// End blob map

// There are often one or more blob maps, hence blob_map_star.
blob_map*
parse_blob_map_star(std::istream &in, const std::string szSection, int &nMaps) {
  getKey(in, szSection, nMaps);
  if(nMaps == 0) {
    std::cerr << "No maps in " << szSection << std::endl;;
    return NULL;
  } else {
    std::cerr << nMaps << " maps in " << szSection << std::endl;;
    blob_map* bm = new blob_map[nMaps];
    for(int i = 0; i < nMaps; i++) 
      parse_blob_map(in, bm[i]);
    return bm;
  }
}

void blob_map_star_print(std::ostream &o, const std::string szSection, blob_map *bm, int nMaps) {
  putKey(o, szSection, nMaps);
  for (int i = 0; i < nMaps; i++ ) blob_map_print(o, bm[i]);
}


class our_file {
public:
  virtual void parse(std::istream &in) = 0;
  virtual void print(std::ostream &o ) = 0;
};

class forward_file : public our_file {
  double loss;
  int nModels, nInputs, nOutputs;
  blob_map *model; // these are the modesl
  blob_map *input;
  blob_map *output;

public:
  void parse(std::istream &in) {
    // getStatement(in, szForward);
    getKey(in, szLoss, loss);
    model  = parse_blob_map_star( in, szModel , nModels );
    input  = parse_blob_map_star( in, szInput , nInputs );
    output = parse_blob_map_star( in, szOutput, nOutputs);
  }

  void print(std::ostream &o) {
    putKey(o, szLoss, loss);
    blob_map_star_print(o, szModel , model , nModels);
    blob_map_star_print(o, szInput , input , nInputs);
    blob_map_star_print(o, szOutput, output, nOutputs);
  }
};

const std::string szMG("MODEL_GRADIENT_BLOBS");
const std::string szIG("INPUT_GRADIENT_BLOBS");

class backward_file : public our_file {
  blob_map *model_g; int nModels; // model gradients
  blob_map *input_g; int nIGs; // input gradients
public:
  void parse(std::istream &in) {
    std::cerr << "parsing backward file" << std::endl;
    model_g = parse_blob_map_star(in, szMG, nModels);
    input_g = parse_blob_map_star(in, szIG, nIGs);
  }
  void print(std::ostream &o) {
    blob_map_star_print(o, szMG , model_g , nModels);
    blob_map_star_print(o, szIG , input_g , nIGs   );
    
  }
};

const std::string szLearn("LEARNING_RATE");
const std::string szReg("REGULARIZATION");
const std::string szGR("global_rate"), szParamRate("param_rate"), szLocalRate("local_rate"), szMomentum("momentum");
const std::string szType("type"), szGlobalRegu("global_regu"), szParamRegu("param_regu"), szLocalRegu("local_regu");



class update_file : public our_file {
  
  double global_rate, param_rate, local_rate, momentum;
  
public:
  void parse(std::istream &in) {
    std::cerr << "parsing update file" << std::endl;
    getStatement(in,szLearn);
    getKey(in, szGR, global_rate);
    getKey(in, szParamRate, param_rate);
    getKey(in, szLocalRate, local_rate);
    getKey(in, szMomentum, momentum);
  }
  void print(std::ostream &o) {
    putStatement(o,szLearn);
    putKey(o, szGR, global_rate);
    putKey(o, szParamRate, param_rate);
    putKey(o, szLocalRate, local_rate);
    putKey(o, szMomentum, momentum);
  }
};
class regularized_update_file : public update_file {
  
  std::string regularization;
  double global_regu, param_regu, local_regu;
  
public:
  void parse(std::istream &in) {
    std::cerr << "parsing regularized update file" << std::endl;
    update_file::parse(in);
    getStatement(in, szReg);
    getKey(in, szType, regularization);
    getKey(in, szGlobalRegu, global_regu);
    getKey(in, szParamRegu, param_regu);
    getKey(in, szLocalRegu, local_regu);
  }
  void print(std::ostream &o) {
    update_file::print(o);
    putStatement(o, szReg);
    putKey(o, szType, regularization);
    putKey(o, szGlobalRegu, global_regu);
    putKey(o, szParamRegu, param_regu);
    putKey(o, szLocalRegu, local_regu);    
  }
};

int tester() {
  std::string fname("test.txt");
  std::ifstream in(fname.c_str());

  // Parse the loss
  double loss = 10.0;
  getKey(in, szLoss, loss);

  std::cout << "got " << loss << std::endl;
  
  // Parse the Model Blob
  int nModels = 20, nInput = 30, nOutput = 40;
  getKey(in, szModel , nModels);
  getKey(in, szInput , nInput);
  getKey(in, szOutput, nOutput);
  
  std::cout << "this many models=" << nModels << std::endl;
  
  // parse the first blob
  blob_map bm;
  parse_blob_map(in, bm);
  
  for (int i = 0; i < bm.nValues; i++) {
    std::cout << "i=" << i <<  " index=" << bm.indexes[i] << " value=" << bm.values[i] << std::endl;
  }
  std::cout << "input=" << nInput << " output=" << nOutput << std::endl;
  in.close();
  return 0;
}


void
process_helper(const std::string fname, our_file &f) {
  std::ifstream i(fname);
  if(i.fail()) { std::cout << "Failed to open file!" << fname << std::endl; exit(-1); }
  f.parse(i);
  f.print(std::cout);
  i.close();
}
int main(){ 
  // forward_file ff;
  // process_helper("../data/sample_snapshot/1424131975000_conv1_1_FORWARD", ff);
  
  // backward_file bf;
  // process_helper("../data/sample_snapshot/1424138428000_conv1_2_BACKWARD", bf);

  // regularized_update_file rf;
  // process_helper("../data/sample_snapshot/1424138428000_conv1_PID0_UPDATE_", rf);

  // update_file uf;
  // process_helper("../data/sample_snapshot/1424138428000_conv1_PID1_UPDATE_", uf);

  forward_file ff1;
  process_helper("../data/sample_snapshot2/1424146592000_conv1_1_FORWARD",ff1);

  forward_file ff2;
  process_helper("../data/sample_snapshot2/1424146793000_relu1_1_FORWARD",ff2);

  backward_file bf1;
  process_helper("../data/sample_snapshot2/1424147161000_relu1_2_BACKWARD", bf1);

  backward_file bf2;
  process_helper("../data/sample_snapshot2/1424147550000_conv1_2_BACKWARD", bf2);

  update_file uf1;
  process_helper("../data/sample_snapshot2/1424147550000_conv1_PID0_UPDATE_", uf1);
  
  return 0;
}
