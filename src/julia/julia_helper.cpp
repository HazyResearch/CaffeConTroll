#include "julia_helper.h"

void Hello(){
	std::cout << "Hi! -- by CaffeConTroll" << std::endl;
}

void ConstructCctNetworkAndRun(uint8_t *solver_pb, int solver_len, uint8_t *net_pb, int net_len){
	BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
	solver_param.ParseFromArray(solver_pb, solver_len);
	net_param.ParseFromArray(net_pb, net_len);
	Corpus * corpus = DeepNet::read_corpus_from_lmdb(net_param, true);
	DeepNet::construct_network(bridges, *corpus, net_param, solver_param);

	Corpus * val_corpus = DeepNet::read_corpus_from_lmdb(net_param, false);

	DeepNet::train_network(bridges, *corpus, net_param, solver_param, "NA", 
		"NA", *val_corpus, false);
}