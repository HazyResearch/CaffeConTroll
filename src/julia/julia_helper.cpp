#include "julia_helper.h"

void Hello(){
	std::cout << "Hi! -- by CaffeConTroll" << std::endl;
}

void train_network(const BridgeVector & bridges, Corpus & corpus, const cnn::NetParameter & net_param,
        const cnn::SolverParameter & solver_param, const string input_model_file, const string snapshot_file_name,
        Corpus & val_corpus, bool time_iterations = false){

	SoftmaxBridge * const softmax = (SoftmaxBridge *) bridges.back();
	Bridge * const first = (Bridge *) bridges.front();

	softmax->p_data_labels->set_p_data(corpus.labels->physical_get_RCDslice(0));
	LogicalCubeFloat * const input_data = first->p_input_layer->p_data_cube;

	float t_load;
	float t_forward;
	float t_backward;
	float t_pass;

	Timer t_total;

#ifdef _LAYER_PROFILING
	const int display_iter = 1;
#else
	const int display_iter = solver_param.display();
#endif
	const int snapshot = solver_param.snapshot();

      // SHADJIS TODO: Support solver_param.test_interval(), i.e. every few training
      // iterations do testing (validation set). For now we can keep the batch size
      // the same during testing but this also does not need to be the case.
	const int test_interval = solver_param.test_interval();

      // Read the number of iterations to run. This is the number of times we will
      // update weights, i.e. the number of mini-batches we will run.
	const size_t num_batch_iterations = solver_param.max_iter();

      // It is necessary to open the reader before loading data to initialize
      // cursor, transaction and environment data
	corpus.OpenLmdbReader();

      // Keep track of the image number in the dataset we are on
	size_t current_image_location_in_dataset = 0;
	size_t current_epoch = 0;    
      // std::cout << "EPOCH: " << current_epoch << std::endl;
	float loss = 0.;
	float accuracy = 0.;

      // Run for max_iter iterations
	for (size_t batch = 0; batch < num_batch_iterations; ++batch) {

		Timer t;
		Timer t2;

        // SHADJIS TODO: corpus.last_batch_size is unused, can remove now
        // SHADJIS TODO: This should be done in parallel with the network execution if slow (measure)
        // SHADJIS TODO: curr_B is unused now in every bridge, can remove it or plan to support variable batch size

        // Read in the next mini-batch from db
		size_t rs = corpus.LoadLmdbData();

        // Note that the implementation of labels has changed.  Since we are reading the lmbd for every
        // iteration, we get the label in the data so now the labels and images objects are parallel
        // TODO? : Perhaps this is a good reason to merge them into a single object 
        // Since labels are in sync they will also only be of size mini_batch_size
		assert(softmax->p_data_labels->get_p_data() == corpus.labels->get_p_data());

        // If we read less than we expected, read the rest from the beginning 
		size_t num_images_left_to_read = corpus.mini_batch_size - rs;
		if (num_images_left_to_read > 0) {
            // Increment epoch
			++current_epoch;
            // Simply reset the cursor so the next load will start from the start of the lmdb
			corpus.ResetCursor();

            // Passing in rs allows us to say that we already filled rs spots in images
            // and now we want to start from that position and complete the set up to mini_batch_size
            // Eg. Minibatch is 10.  We read 2 images and hit the end of the mldb.  After reseting the
            // cursor above we can just tell the load function to start from index 2 and continue
			size_t rs2 = corpus.LoadLmdbData(rs);
			assert(rs2 == num_images_left_to_read);

            // The corpus.labels object was also updated above so we need to check that
            // the pointer is still consistent
			assert(softmax->p_data_labels->get_p_data() == corpus.labels->get_p_data());
		}

		current_image_location_in_dataset += corpus.mini_batch_size;
		if (current_image_location_in_dataset >= corpus.n_images) {
			current_image_location_in_dataset -= corpus.n_images;
		}

		t_load = t.elapsed();

		t.restart();
        // initialize input_data for this mini batch
        // Ce: Notice the change here compared with the master branch -- this needs to be refactored
        // to make the switching between this and the master branch (that load everything in memory)
        // dynamically and improve code reuse.
		float * const mini_batch = corpus.images->physical_get_RCDslice(0);
		assert(input_data->get_p_data() == mini_batch);

		softmax->reset_loss();

        // forward pass
		DeepNet::run_forward_pass(bridges);

		t_forward = t.elapsed();

		loss += (softmax->get_loss() / float(corpus.mini_batch_size));
		accuracy += float(DeepNet::find_accuracy(softmax->p_data_labels, (*--bridges.end())->p_output_layer->p_data_cube)) / float(corpus.mini_batch_size);

        // backward pass
		t.restart();
		DeepNet::run_backward_pass(bridges);
		t_backward = t.elapsed();

		t_pass = t2.elapsed();

        // Check if we should print batch status
        // Edit: Instead we will make display_iter print the average since
        // the previous display, since this seems more useful
		if ( (batch+1) % display_iter == 0 ) {
			float learning_rate = Util::get_learning_rate(solver_param.lr_policy(), solver_param.base_lr(), solver_param.gamma(),
				batch+1, solver_param.stepsize(), solver_param.power(), solver_param.max_iter());

			std::cout << "Training Status Report (Epoch " << current_epoch << " / Mini-batch iter " << batch << "), LR = " << learning_rate << std::endl;
			std::cout << "  \033[1;32m";
			std::cout << "Loss & Accuracy [Average of Past " << display_iter << " Iterations]\t" << loss/float(display_iter) << "\t" << float(accuracy)/(float(display_iter));
			std::cout << "\033[0m" << std::endl;
			loss = 0.;
			accuracy = 0.;

			if (time_iterations) {
				std::cout << "\033[1;31m";
				std::cout << "  Iteration Time Status Report (seconds)" << std::endl;
				std::cout << "    Loading Data:  " << t_load << std::endl;
				std::cout << "    Forward Pass:  " << t_forward << std::endl;
				std::cout << "    Backward Pass: " << t_backward << std::endl;
				std::cout << "    Total:         " << t_pass << std::endl;
				std::cout << "\033[0m";
			}

		}
        // Check if we should run validation
		if (test_interval > 0 && (batch+1) % test_interval == 0) {
			std::cout << "Validation/Test Status Report (Epoch " << current_epoch << " / Mini-batch iter " << batch << ")" << std::endl;
            // Switch dataset to val
			std::cout << "  \033[1;36m";
			bridges[0]->update_p_input_layer_data_CPU_ONLY(val_corpus.images->physical_get_RCDslice(0));
			DeepNetConfig::setTrain(false);
			DeepNet::test_network(bridges, val_corpus, net_param, solver_param, time_iterations);
            // Switch dataset back to train
            // reset the softmax data labels to the corpus labels instead of the test labels
			softmax->p_data_labels->set_p_data(corpus.labels->physical_get_RCDslice(0));
			bridges[0]->update_p_input_layer_data_CPU_ONLY(corpus.images->physical_get_RCDslice(0));
			DeepNetConfig::setTrain(true);
			std::cout << "    [Run on entire validation set]\033[0m" << std::endl;
		}
        // Check if we should write a snapshot
		if (snapshot > 0 && (batch+1) % snapshot == 0) {
			time_t rawtime;
			struct tm * timeinfo;
			char buffer[80];
			time (&rawtime);
			timeinfo = localtime(&rawtime);
			strftime(buffer,80,"%d-%m-%Y-%I-%M-%S",timeinfo);
			std::string str(buffer);
			std::string snapshot_name;

			if (snapshot_file_name == "NA") {
				snapshot_name = "trained_model.bin." + str;
			} else {
				snapshot_name = snapshot_file_name + "." + str;
			}
			DeepNet::write_model_to_file(bridges, snapshot_name);
			std::cout << "======= Writing snapshot " << snapshot_name << " =======\n";
		}
	}

      // This frees any handles we have to the lmdb and free allocated internal objects.
      // Note that corpus.images and corpus.labels are still usable
	corpus.CloseLmdbReader();
	std::cout << "Total Time (seconds): " << t_total.elapsed() << std::endl;
}

void ConstructCctNetworkAndRun(uint8_t *solver_pb, int solver_len, uint8_t *net_pb, int net_len){
	BridgeVector bridges; cnn::SolverParameter solver_param; cnn::NetParameter net_param;
	solver_param.ParseFromArray(solver_pb, solver_len);
	net_param.ParseFromArray(net_pb, net_len);
	Corpus * corpus = DeepNet::read_corpus_from_lmdb(net_param, true);
	DeepNet::construct_network(bridges, *corpus, net_param, solver_param);

	Corpus * val_corpus = DeepNet::read_corpus_from_lmdb(net_param, false);

	train_network(bridges, *corpus, net_param, solver_param, "NA", 
		"NA", *val_corpus, false);

	// for (Bridge* bridge : bridges){
	// 	std::cout << "The Name::::: -> " << bridge->name << std::endl;
	// 	std::cout << "input" << std::endl;
	// 	bridge->p_input_layer->p_data_cube->logical_print();
	// 	std::cout << "output" << std::endl;
	// 	bridge->p_output_layer->p_data_cube->logical_print();
	// 	std::cout << "input grad" << std::endl;
	// 	bridge->p_input_layer->p_gradient_cube->logical_print();
	// 	std::cout << "output grad" << std::endl;
	// 	bridge->p_output_layer->p_gradient_cube->logical_print();
	// }
}

typedef struct NetworkMetadata{
	BridgeVector bridges;
	cnn::SolverParameter solver_param;
	cnn::NetParameter net_param;
	Corpus * corpus;
	Corpus * val_corpus;
	size_t batch = 0;
	//SoftmaxBridge * softmax;
	Bridge * first;
	float loss = 0.0;
	float accuracy = 0.0;

} network_t;

void AugmentIteration(network_t *net, string snapshot_file_name){
	size_t batch = net->batch;
	const int display_iter = net->solver_param.display();
	const int snapshot = net->solver_param.snapshot();
	const int test_interval = net->solver_param.test_interval();

	    // Check if we should print batch status
        // Edit: Instead we will make display_iter print the average since
        // the previous display, since this seems more useful
	if ( (batch+1) % display_iter == 0 ) {
		float learning_rate = Util::get_learning_rate(net->solver_param.lr_policy(), net->solver_param.base_lr(), net->solver_param.gamma(),
			batch+1, net->solver_param.stepsize(), net->solver_param.power(), net->solver_param.max_iter());

		std::cout << "Training Status Report ( Mini-batch iter " << batch << "), LR = " << learning_rate << std::endl;
		std::cout << "  \033[1;32m";
		std::cout << "Loss & Accuracy [Average of Past " << display_iter << " Iterations]\t" << net->loss/float(display_iter) << "\t" << float(net->accuracy)/(float(display_iter));
		std::cout << "\033[0m" << std::endl;
        net->loss = 0.;
     	net->accuracy = 0.;

	}
    // Check if we should run validation
	if (test_interval > 0 && (batch+1) % test_interval == 0) {
		std::cout << "Validation/Test Status Report (Epoch Mini-batch iter " << batch << ")" << std::endl;
            // Switch dataset to val
		std::cout << "  \033[1;36m";
		net->bridges[0]->update_p_input_layer_data_CPU_ONLY(net->val_corpus->images->physical_get_RCDslice(0));
		DeepNetConfig::setTrain(false);
		DeepNet::test_network(net->bridges, *(net->val_corpus), net->net_param, net->solver_param, false);
        // Switch dataset back to train
        // reset the softmax data labels to the corpus labels instead of the test labels
		net->bridges[0]->update_p_input_layer_data_CPU_ONLY(net->corpus->images->physical_get_RCDslice(0));
		DeepNetConfig::setTrain(true);
		std::cout << "    [Run on entire validation set]\033[0m" << std::endl;
	}
    // Check if we should write a snapshot
	if (snapshot > 0 && (batch+1) % snapshot == 0) {
		time_t rawtime;
		struct tm * timeinfo;
		char buffer[80];
		time (&rawtime);
		timeinfo = localtime(&rawtime);
		strftime(buffer,80,"%d-%m-%Y-%I-%M-%S",timeinfo);
		std::string str(buffer);
		std::string snapshot_name;

		if (snapshot_file_name == "NA") {
			snapshot_name = "trained_model.bin." + str;
		} else {
			snapshot_name = snapshot_file_name + "." + str;
		}
		DeepNet::write_model_to_file(net->bridges, snapshot_name);
		std::cout << "======= Writing snapshot " << snapshot_name << " =======\n";
	}
}

void* InitNetwork(uint8_t *solver_pb, int solver_len, uint8_t *net_pb, int net_len, char* model_file){
	network_t* net = new network_t;
	net->solver_param.ParseFromArray(solver_pb, solver_len);
	net->net_param.ParseFromArray(net_pb, net_len);
	//std::cout << net->net_param.DebugString() << std::endl;
	//Parser::read_proto_from_text_file("julia-pb/lenet_solver.prototxt", &(net->solver_param));
    //Parser::read_net_params_from_text_file("julia-pb/lenet_train_test.prototxt", &(net->net_param));
	net->corpus = DeepNet::read_corpus_from_lmdb(net->net_param, true);
	DeepNet::construct_network(net->bridges, *(net->corpus), net->net_param, net->solver_param);
	net->val_corpus = DeepNet::read_corpus_from_lmdb(net->net_param, false);
	std::string model = std::string(model_file);
	if (model != "NA"){
		DeepNet::read_model_from_file(net->bridges, model);
	}
	//net->softmax = (SoftmaxBridge *) net->bridges.back();
	net->first = (Bridge *) net->bridges.front();
	//net->softmax->p_data_labels->set_p_data(net->corpus->labels->physical_get_RCDslice(0));

	net->corpus->OpenLmdbReader();
	return net;
}

void _SingleForwardPass(void *_net, const char **keys=NULL, int key_size=0){
	// Calling from Julia.  Convert net to network_t
	network_t *net = (network_t *)_net;
	cnn::SolverParameter solver_param = net->solver_param;
	cnn::NetParameter net_param = net->net_param;
	Corpus& corpus = *net->corpus;
	Corpus& val_corpus = *net->val_corpus;
	const string snapshot_file_name = "NA";

	LogicalCubeFloat * const input_data = net->first->p_input_layer->p_data_cube;

	size_t rs = 0;
	if (keys){
		rs = corpus.LoadLmdbData(keys, key_size, 0);
	} else {
		rs = corpus.LoadLmdbData();
	}

        // Note that the implementation of labels has changed.  Since we are reading the lmbd for every
        // iteration, we get the label in the data so now the labels and images objects are parallel
        // TODO? : Perhaps this is a good reason to merge them into a single object 
        // Since labels are in sync they will also only be of size mini_batch_size
	//assert(net->softmax->p_data_labels->get_p_data() == corpus.labels->get_p_data());

        // If we read less than we expected, read the rest from the beginning 
	size_t num_images_left_to_read = net->corpus->mini_batch_size - rs;
	if (num_images_left_to_read > 0) {
            // Simply reset the cursor so the next load will start from the start of the lmdb
		corpus.ResetCursor();

            // Passing in rs allows us to say that we already filled rs spots in images
            // and now we want to start from that position and complete the set up to mini_batch_size
            // Eg. Minibatch is 10.  We read 2 images and hit the end of the mldb.  After reseting the
            // cursor above we can just tell the load function to start from index 2 and continue
		assert(!keys); // if we are using keys we should never be given less than a mini batch
		size_t rs2 = corpus.LoadLmdbData(rs);
		assert(rs2 == num_images_left_to_read);

            // The corpus.labels object was also updated above so we need to check that
            // the pointer is still consistent
		//assert(net->softmax->p_data_labels->get_p_data() == corpus.labels->get_p_data());
	}
    // initialize input_data for this mini batch
    // Ce: Notice the change here compared with the master branch -- this needs to be refactored
    // to make the switching between this and the master branch (that load everything in memory)
    // dynamically and improve code reuse.
	float * const mini_batch = net->corpus->images->physical_get_RCDslice(0);
	assert(input_data->get_p_data() == mini_batch);

	//net->softmax->reset_loss();

    // forward pass
	DeepNet::run_forward_pass(net->bridges);
	//net->bridges[7]->p_output_layer->p_data_cube->logical_print();


	//net->loss += (net->softmax->get_loss() / float(net->corpus->mini_batch_size));
	//net->accuracy += float(DeepNet::find_accuracy(net->softmax->p_data_labels, (*--(net->bridges.end()))->p_output_layer->p_data_cube)) / float(net->corpus->mini_batch_size);

	AugmentIteration(net, snapshot_file_name);
	net->batch++;
}

void SingleForwardPassData(void *_net, char **keys, int key_size){
	_SingleForwardPass(_net, (const char **)keys, key_size);
}

void SingleForwardPass(void *_net){
	_SingleForwardPass(_net);
}

void SingleBackwardPass(void *_net){
	// Calling from Julia.  Convert net to network_t
	network_t *net = (network_t *)_net;

	// backward pass
	DeepNet::run_backward_pass(net->bridges);
	//net->bridges.back()->p_output_layer->p_gradient_cube->logical_print();
}

void DeleteNetwork(void *_net){
	network_t *net = (network_t *)_net;
	net->corpus->CloseLmdbReader();
	free(net->corpus);
	free(net->val_corpus);
	free(net);
}

void GetScores(void *_net, float *scores){
	network_t *net = (network_t *)_net;
	Bridge * last = net->bridges.back();
	//last->p_output_layer->p_data_cube->logical_print();
	memcpy(scores, last->p_output_layer->p_data_cube->get_p_data(), 
		last->p_output_layer->p_data_cube->n_elements * sizeof(float));
}

void GetWeights(void *_net, float *weights, int layer_index){
	network_t *net = (network_t *)_net;
	Bridge * layer = net->bridges.at(layer_index);
	//last->p_output_layer->p_data_cube->logical_print();
	memcpy(weights, layer->p_output_layer->p_data_cube->get_p_data(), 
		layer->p_output_layer->p_data_cube->n_elements * sizeof(float));
}

void SetGradient(void *_net, float *dscores){
	network_t *net = (network_t *)_net;
	Bridge * last = net->bridges.back();
	//last->p_output_layer->p_data_cube->logical_print();
	memcpy(last->p_output_layer->p_gradient_cube->get_p_data(), dscores,
		last->p_output_layer->p_gradient_cube->n_elements * sizeof(float));
	//last->p_output_layer->p_gradient_cube->logical_print();
}

void GetLabels(void *_net, float *labels){
	network_t *net = (network_t *)_net;
	//net->softmax->p_data_labels->logical_print();
	memcpy(labels, net->corpus->labels->get_p_data(),
		net->corpus->labels->n_elements * sizeof(float));
}






















