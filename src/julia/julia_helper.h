#ifndef _JULIA_HELPER_H
#define _JULIA_HELPER_H

#include "../DeepNet.h"

extern "C" {

	void Hello();

	void ConstructCctNetworkAndRun(uint8_t *solver_pb, int solver_len, uint8_t *net_pb, int net_len);

	void* InitNetwork(uint8_t *solver_pb, int solver_len, uint8_t *net_pb, int net_len, char* model_file);

	void SingleForwardPass(void *_net);

	void SingleForwardPassData(void *_net, char **keys, int key_size);

	void SingleForwardPassAddition(void *_net, int index, void* input_data);

	void SingleBackwardPass(void *_net);

	void DeleteNetwork(void *_net);

	void GetScores(void *_net, float *scores);

	// TODO : Delete this!  This is just a hack to test softmax with internal labels
	// Figure out how to move labels outside
	void GetLabels(void *_net, float *labels);

	void SetGradient(void *_net, float *dscores);

	void GetWeights(void *_net, float *weights, int layer_index);

	void PrintLayerOutputSize(void *_net, int index);

	void* GetOutputDataPointer(void *_net, int index);

	void* GetGradientPointer(void *_net, int index);

	void PrintLayerOutput(void *_net, int index);

	void PrintLayerInput(void *_net, int index);

	void PrintLayerGradient(void *_net, int index);
}

#endif