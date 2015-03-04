
#include "DeviceDriver.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void _sapply(float * dst, int numElements, FUNC_STRANSFORM func, 
						void * const func_curry){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < numElements){
		dst[i] = (*func)(dst[i], func_curry);
	}
}