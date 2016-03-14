#ifndef _JULIA_HELPER_H
#define _JULIA_HELPER_H

#include "../DeepNet.h"

extern "C" {

	void Hello();

	void ConstructCctNetworkAndRun(uint8_t *solver_pb, int solver_len, uint8_t *net_pb, int net_len);

}

#endif