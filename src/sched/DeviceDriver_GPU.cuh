
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


__global__ void _sreduce(float * dst, int numElements, float * src1, float * src2,
						FUNC_SREDUCE func, void * const func_curry){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < numElements){
		dst[i] = (*func)(src1[i], src2[i], func_curry);
	}
}

__global__ void _spmap(float * dst, float * src, int numElements, int srcSkip,
	FUNC_IDX_MAPPING idx_func, void * const idx_func_curry,
	FUNC_MM_MAPPING func, void * const func_curry){

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	i = i * srcSkip;
	int src_idx, dst_idx;

	for(int j=0; j<srcSkip; j++){
		src_idx = i + j;
		if(src_idx < numElements){
			dst_idx = (*idx_func)(src_idx, idx_func_curry);
			(*func)(&dst[dst_idx], &src[src_idx], func_curry);
		}
	}

}

