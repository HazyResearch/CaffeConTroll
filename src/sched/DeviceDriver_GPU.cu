
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

#include "DeviceDriver.h"
#include "DeviceDriver_GPU.h"

#include "../kernels/include.hxx"


__host__ __device__ float __sconstant_initialize_helper(float a, void * arg){
  return *((float*)arg);
}

template<FUNC_STRANSFORM func>
__global__ void _sapply(float * dst, int numElements, void * const func_curry){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < numElements){
    dst[i] = func(dst[i], func_curry);
  }
}

template<FUNC_SREDUCE func>
__global__ void _sreduce(float * dst, int numElements, float * src1, float * src2, 
	void * const func_curry){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < numElements){
    dst[i] = func(src1[i], src2[i], func_curry);
  }
}

template<FUNC_IDX_MAPPING idx_func, FUNC_MM_MAPPING func>
__global__ void _spmap(float * dst, float * src, int numElements, int srcSkip,
  void * const idx_func_curry, void * const func_curry){

  int i = (blockDim.x * blockIdx.x + threadIdx.x);
  i = i * srcSkip;
  int src_idx, dst_idx;

  src_idx = i;
  if(src_idx < numElements*srcSkip){
    dst_idx = idx_func(src_idx, idx_func_curry);
    func(&dst[dst_idx/sizeof(float)], &src[src_idx/sizeof(float)], func_curry);
  }
}

template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
__global__ void _spmap_readc(float* dst, float * src, PMapHelper args){
	const size_t block_x = blockIdx.x;
	const size_t block_y = blockIdx.y;

	//const size_t nRblock = args.sR/args.sBR;
	const size_t nCblock = args.sC/args.sBC;

	Block2D input_block;
	input_block.r = block_x / nCblock;
	input_block.c = block_x % nCblock;
	input_block.d = block_y % args.sD;
	input_block.b = block_y / args.sD;
	input_block.dr = args.sR;
	input_block.dc = args.sC;

	Block2D output_block;
	f_id(&output_block, &input_block, &args);

	const size_t datar = threadIdx.y + input_block.r * args.sR;
	const size_t datac = threadIdx.x + input_block.c * args.sC;

	PointIn2DBlock point;
	point.block = input_block;
	point.data = src[
		args.sR * args.sC * (args.sD * input_block.b + input_block.d) +
		datar + args.sC +
		datac
	];
	point.r = threadIdx.y;
	point.c = threadIdx.x;

	f_data(dst, &output_block, &point, &args);

}


template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
void GPUDriver::pmap2d_read_coalesce(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct PMapHelper args){

	// input block sizes
	size_t sBR = args.sBR, sBC = args.sBC;
	dim3 threadsPerBlock(sBC, sBR);	// trivial impl -- each input pixel is a single thread
	dim3 numBlocks(args.sR*args.sC/sBC/sBR, args.sD*args.sB);

	_spmap_readc<f_id,f_data><<<numBlocks, threadsPerBlock>>>((float*) dst->ptr, (float*) src->ptr, args);
	err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _spmap_readc"  << "  ERROR " << err << std::endl;
	  assert(false);
	}
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to cudaDeviceSynchronize _spmap_readc"  << "  ERROR " << err << std::endl;
	  assert(false);
	}

}


GPUDriver::GPUDriver(){
    cublasCreate(&handle);
}

DeviceMemoryPointer * GPUDriver::get_device_pointer(void * ptr, size_t size_in_byte){
	// TODO: This has memory leak! Refactor it!
	return new DeviceMemoryPointer_Local_GPURAM(gpu_id, ptr, size_in_byte);
}

void GPUDriver::malloc(DeviceMemoryPointer * dst){
	cudaMalloc((void**)&dst->ptr, dst->size_in_byte);
}

void GPUDriver::free(DeviceMemoryPointer * dst){
	cudaFree(dst->ptr);
}

void GPUDriver::memcpy(DeviceMemoryPointer * dst, DeviceMemoryPointer * src){
	#ifdef _DO_ASSERT
	assert(dst->size_in_byte == src->size_in_byte);
	#endif
	if(src->type == DEVICEMEMORY_LOCAL_RAM){
  		cudaMemcpy(dst->ptr, src->ptr, dst->size_in_byte, cudaMemcpyHostToDevice);
	}else if(dst->type == DEVICEMEMORY_LOCAL_RAM){
  		cudaMemcpy(dst->ptr, src->ptr, dst->size_in_byte, cudaMemcpyDeviceToHost);
	}else{
		cudaMemcpy(dst->ptr, src->ptr, dst->size_in_byte, cudaMemcpyDeviceToDevice);
	}
}

void GPUDriver::memset(DeviceMemoryPointer * dst, const char value){
	#ifdef _DO_ASSERT
	assert(dst->type==DEVICEMEMORY_LOCAL_RAM);
	#endif
	cudaMemset(dst->ptr, value, dst->size_in_byte);
}

template<FUNC_IDX_MAPPING f_dst_pos, FUNC_MM_MAPPING func>
void GPUDriver::parallel_map(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry, DeviceMemoryPointer * const func_curry){

	// create a device version of func_curry
	void * d_func_curry;
	cudaMalloc((void**)&d_func_curry, func_curry->size_in_byte);
	cudaMemcpy(d_func_curry, func_curry->ptr, func_curry->size_in_byte, cudaMemcpyHostToDevice);

	void * d_idx_func_curry;
	cudaMalloc((void**)&d_idx_func_curry, f_dst_pos_curry->size_in_byte);
	cudaMemcpy(d_idx_func_curry, f_dst_pos_curry->ptr, f_dst_pos_curry->size_in_byte, cudaMemcpyHostToDevice);

	// Run.
	const int n_elements =  dst->size_in_byte / src_skip;
	int blocksPerGrid = (n_elements + 1 + threadsPerBlock - 1) / threadsPerBlock;
	_spmap<f_dst_pos,func><<<blocksPerGrid, threadsPerBlock>>>((float*) dst->ptr, (float *) src->ptr,
	  n_elements, src_skip, d_idx_func_curry, d_func_curry);
	err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _spmap"  << "  ERROR " << err << std::endl;
	  assert(false);
	}
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to sync _spmap"  << "  ERROR " << err << std::endl;
	  assert(false);
	}

	cudaFree(d_func_curry);
	cudaFree(d_idx_func_curry);

}

void GPUDriver::smath_axpy(const float alpha, DeviceMemoryPointer * X, DeviceMemoryPointer * Y)  { 
#ifdef _DO_ASSERT
	assert(X->type==DEVICEMEMORY_LOCAL_RAM);
	assert(Y->type==DEVICEMEMORY_LOCAL_RAM);
	assert(X->size_in_byte==Y->size_in_byte);
#endif
  int n_elements = X->size_in_byte / sizeof(float);
  status = cublasSaxpy(handle, n_elements, &alpha, (float*)X->ptr, 1, (float*)Y->ptr, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);
}

template<FUNC_STRANSFORM func>
void GPUDriver::sapply(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry){
	#ifdef _DO_ASSERT
	assert(dst->type==DEVICEMEMORY_LOCAL_RAM);
	assert(dst->size_in_byte % sizeof(float) == 0);
	#endif
	// TODO: Refactoring

	// Second, create a device version of func_curry
	void * d_func_curry;
	cudaMalloc((void**)&d_func_curry, func_curry->size_in_byte);
	cudaMemcpy(d_func_curry, func_curry->ptr, func_curry->size_in_byte, cudaMemcpyHostToDevice);

	// Run.
	const int n_elements =  dst->size_in_byte / sizeof(float);
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;
	_sapply<func><<<blocksPerGrid, threadsPerBlock>>>((float*) dst->ptr, n_elements, d_func_curry);
	err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _sapply" << "  ERROR " << err << std::endl;
	  assert(false);
	}
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	assert(err == cudaSuccess);

	cudaFree(d_func_curry);
}

void GPUDriver::smath_axpby(const float alpha, DeviceMemoryPointer * X, const float beta, DeviceMemoryPointer * Y) { 
#ifdef _DO_ASSERT
  assert(X->size_in_byte == Y->size_in_byte);
  assert(X->size_in_byte % sizeof(float) == 0);
#endif

  int n_elements = X->size_in_byte / sizeof(float);
  status = cublasSscal(handle, n_elements, &beta, (float*)Y->ptr, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);

  status = cublasSaxpy(handle, n_elements, &alpha, (float*)X->ptr, 1, (float*)Y->ptr, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);

}

void GPUDriver::set_num_threads(const int nThreads) { 
}


void GPUDriver::sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
    int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
    float beta, float * pC, int LDC){
  
	if(TA == CblasNoTrans && TB == CblasNoTrans){

		cublasOperation_t ta = CUBLAS_OP_N;
		cublasOperation_t tb = CUBLAS_OP_N;

		status = cublasSgemm(handle, tb, ta, N, M, K, &alpha, 
			pB, N, pA, K, &beta, pC, N); 

		cudaDeviceSynchronize();
		err = cudaGetLastError();
		assert(err == cudaSuccess);

		assert(status == CUBLAS_STATUS_SUCCESS);

	}else{
		assert(false);
	}

}

template<FUNC_SREDUCE func>
void GPUDriver::selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1, 
DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry){ 

	#ifdef _DO_ASSERT
	assert(dst->size_in_byte == src1->size_in_byte);
	assert(dst->size_in_byte == src2->size_in_byte);
	assert(dst->size_in_byte % sizeof(float) == 0);
	#endif

	// create a device version of func_curry
	void * d_func_curry;
	cudaMalloc((void**)&d_func_curry, func_curry->size_in_byte);
	cudaMemcpy(d_func_curry, func_curry->ptr, func_curry->size_in_byte, cudaMemcpyHostToDevice);

	// Run.
	const int n_elements =  dst->size_in_byte / sizeof(float);
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;
	_sreduce<func><<<blocksPerGrid, threadsPerBlock>>>((float*) dst->ptr, n_elements, 
	  (float*) src1->ptr, (float*) src2->ptr, d_func_curry);
	err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _sreduce" << std::endl;
	  assert(false);
	}
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	assert(err == cudaSuccess);


}

/**
* This function is called only once. So its speed does not matter.
* TODO: Wrap this up with CURAND.
**/
void GPUDriver::sinitialize_xavier(DeviceMemoryPointer *arr, const size_t n_batch) {
	const size_t n_arr_elements = arr->size_in_byte / sizeof(float);
	const size_t fan_in = n_arr_elements / n_batch;
	const float scale = sqrt(3.0 / fan_in);

	mt19937 gen(rd());
	uniform_real_distribution<float> uni(-scale, scale);
	float * temp = new float[n_arr_elements];
	for(int i=0;i<n_arr_elements;i++){
	  temp[i] = uni(gen);
	}
	cudaMemcpy(arr->ptr, temp, arr->size_in_byte, cudaMemcpyHostToDevice);
	delete[] temp;
	}

/**
* This function is called only once. So its speed does not matter.
* TODO: Wrap this up with CURAND.
**/
void GPUDriver::sbernoulli_initialize(DeviceMemoryPointer *arr, const float p) {
const size_t n_arr_elements = arr->size_in_byte / sizeof(float);

	mt19937 gen(rd());
	bernoulli_distribution bern(p);
	float * temp = new float[n_arr_elements];
	for(int i=0;i<n_arr_elements;i++){
	  temp[i] = bern(gen);
	}
	cudaMemcpy(arr->ptr, temp, arr->size_in_byte, cudaMemcpyHostToDevice);
	delete[] temp;

}

/**
* This function is called only once. So its speed does not matter.
* TODO: Wrap this up with CURAND.
**/
void GPUDriver::sgaussian_initialize(DeviceMemoryPointer *arr, const float mean, const float std_dev) {
const size_t n_arr_elements = arr->size_in_byte / sizeof(float);
	mt19937 gen(rd());
	normal_distribution<float> gaussian(mean, std_dev);
	float * temp = new float[n_arr_elements];
	for(int i=0;i<n_arr_elements;i++){
	  temp[i] = gaussian(gen);
	}
	cudaMemcpy(arr->ptr, temp, arr->size_in_byte, cudaMemcpyHostToDevice);
	delete[] temp;

}

void GPUDriver::sconstant_initialize(DeviceMemoryPointer *arr, const float value){
    DeviceMemoryPointer_Local_RAM pvalue((void*)&value, sizeof(float));
    sapply<__sconstant_initialize_helper>(arr, &pvalue);
}

void * GPUDriver::choose_ptr(void * host, void * device){
	return device;
}

/**
 * This is necessary for template to be instantiated.
 */
template void GPUDriver::pmap2d_read_coalesce<_fpmap_id,_fmap_lower>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src, const struct PMapHelper args);

template void GPUDriver::pmap2d_read_coalesce<_fpmap_id,_fmap_remap>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src, const struct PMapHelper args);

/** All template instantiations for parallel_map **/
template void GPUDriver::parallel_map<_f_idx_strid4_copy,_f_strid4_copy>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Conv/FC Bias forward
template void GPUDriver::parallel_map<_f_src_to_dst_bias_forward,_f_bias_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// ReLU forward
template void GPUDriver::parallel_map<_f_src_to_dst_relu_forward,_f_relu_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// ReLU backward
template void GPUDriver::parallel_map<_f_src_to_dst_relu_backward,_f_relu_backward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Dropout forward train
template void GPUDriver::parallel_map<_f_src_to_dst_dropout_forward,_f_dropout_forward_train>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Dropout forward test
template void GPUDriver::parallel_map<_f_src_to_dst_dropout_forward,_f_dropout_forward_test>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Pool forward
template void GPUDriver::parallel_map<_f_src_to_dst_pool_forward,_f_pool_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Pool backward
template void GPUDriver::parallel_map<_f_src_to_dst_pool_backward,_f_pool_backward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// LRN forward
template void GPUDriver::parallel_map<_f_src_to_dst_lrn_forward,_f_lrn_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// LRN forward normalize
template void GPUDriver::parallel_map<_f_src_to_dst_lrn_forward,_f_lrn_forward_normalize>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// LRN backward
template void GPUDriver::parallel_map<_f_src_to_dst_lrn_backward,_f_lrn_backward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Softmax forward
template void GPUDriver::parallel_map<_f_src_to_dst_softmax_forward,_f_softmax_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Softmax backward
template void GPUDriver::parallel_map<_f_src_to_dst_softmax_backward,_f_softmax_backward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);

template void GPUDriver::sapply<_f_add_one>(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry);

template void GPUDriver::sapply<_f_set>(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry);

template void GPUDriver::selementwise_reduce2<_f_reduce>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src1, DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);

template void GPUDriver::selementwise_reduce2<_f_reduce_mul>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src1, DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);

template void GPUDriver::selementwise_reduce2<_f_reduce_tanhgrad>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src1, DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);

