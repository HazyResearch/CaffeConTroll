
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include "src/kernels/lowering.hxx"

#include "DeviceDriver.h"
#include "DeviceDriver_GPU.h"

__host__ __device__ float __sconstant_initialize_helper(float a, void * arg){
  return *((float*)arg);
}
__device__ FUNC_STRANSFORM _sconstant_initialize_helper = __sconstant_initialize_helper;

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


__device__ 
void _fpmap_id2(Block2D * const output_block, const Block2D * const input_block, const PMapHelper * const args){
	output_block->r = 0;
	output_block->c = 0;
	output_block->d = 0;
	output_block->d = 0;
	output_block->dr = args->kR;
	output_block->dc = args->kC;
}

__device__ 
void _fmap_lower2(float * output, const Block2D * const output_block, const PointIn2DBlock * const input_point, const PMapHelper * const args){
	
	const size_t ir = input_point->r;
	const size_t ic = input_point->c;
	const size_t ib = input_point->block.b;
	const size_t id = input_point->block.d;

	const size_t kR = args->kR;
	const size_t kC = args->kC;
	const size_t iR = args->sR;
	const size_t iC = args->sC;
	const size_t iB = args->sB;
	const size_t o_base_col = ib * (iR-kR+1)*(iC-kC+1);
	const size_t o_base_row = id * kR * kC;
	const size_t oC = iB * (iR-kR+1)*(iC-kC+1);

	const float input = input_point->data;

	for(int r=ir-kR;r<=ir;r++){
		int dr = ir-r;
		for(int c=ic-kC;c<=ic;c++){
			int dc = ic-c;
			int ocol = r*iC+c;
			int orow = dr*kC+dc;
			int ocol2 = ocol + o_base_col;
			int orow2 = orow + o_base_row;
			// then write to ocol, orow
			if(ocol >= 0 && ocol < (iR-kR+1)*(iC-kC+1) && orow >= 0 && orow < kR*kC){
				output[ocol + orow*oC] = input;
			}
		}
	}
}

template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
__global__ void _spmap_readc(float* dst, float * src, PMapHelper args){
	const size_t block_x = blockIdx.x;
	const size_t block_y = blockIdx.y;

	const size_t nRblock = args.sR/args.sBR;
	const size_t nCblock = args.sC/args.sBC;

	Block2D input_block;
	input_block.r = block_x / nCblock;
	input_block.c = block_x % nCblock;
	input_block.d = block_y % args.sD;
	input_block.b = block_y / args.sD;
	input_block.dr = args.sR;
	input_block.dc = args.sC;

	Block2D output_block;
	//(*f_id)(&output_block, &input_block, &args);
	//_fpmap_id2(&output_block, &input_block, &args);
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

	//(*f_data)(dst, &output_block, &point, &args);
	//_fmap_lower2(dst, &output_block, &point, &args);
	f_data(dst, &output_block, &point, &args);

}


template<__device__ FPMAP_ID f_id, __device__ FPMAP_DATA_READC f_data>
void GPUDriver::pmap2d_read_coalesce(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct PMapHelper args){

	// First, create host version of func
	//FPMAP_DATA_READC h_func;
	//cudaMemcpyFromSymbol(&h_func, *f_data, sizeof(FPMAP_DATA_READC));
	//FPMAP_DATA_READC d_myfunc = h_func;

	//FPMAP_ID h_idx_func;
	//cudaMemcpyFromSymbol(&h_idx_func, *f_id, sizeof(FPMAP_ID));
	//FPMAP_ID d_idx_myfunc = h_idx_func;

	// input block sizes
	size_t sBR = args.sBR, sBC = args.sBC;
	dim3 threadsPerBlock(sBC, sBR);	// trivial impl -- each input pixel is a single thread
	dim3 numBlocks(args.sR*args.sC/sBC/sBR, args.sD*args.sB);

	//_spmap_readc<_fpmap_id,_fmap_lower><<<numBlocks, threadsPerBlock>>>((float*) dst->ptr, (float*) src->ptr, args);

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
	assert(dst->type==DEVICEMEMORY_LOCAL_RAM);
	assert(src->type==DEVICEMEMORY_LOCAL_RAM);
	assert(dst->size_in_byte == src->size_in_byte);
	#endif
	cudaMemcpy(dst->ptr, src->ptr, dst->size_in_byte, cudaMemcpyDeviceToDevice);
}

void GPUDriver::memset(DeviceMemoryPointer * dst, const char value){
	#ifdef _DO_ASSERT
	assert(dst->type==DEVICEMEMORY_LOCAL_RAM);
	#endif
	cudaMemset(dst->ptr, value, dst->size_in_byte);
}

void GPUDriver::parallel_map(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
size_t src_skip, FUNC_IDX_MAPPING * f_dst_pos, DeviceMemoryPointer * const f_dst_pos_curry,
FUNC_MM_MAPPING * func, DeviceMemoryPointer * const func_curry){

	// First, create host version of func
	FUNC_MM_MAPPING h_func;
	cudaMemcpyFromSymbol(&h_func, *func, sizeof(FUNC_MM_MAPPING));
	FUNC_MM_MAPPING d_myfunc = h_func;

	FUNC_IDX_MAPPING h_idx_func;
	cudaMemcpyFromSymbol(&h_idx_func, *f_dst_pos, sizeof(FUNC_IDX_MAPPING));
	FUNC_IDX_MAPPING d_idx_myfunc = h_idx_func;

	// Second, create a device version of func_curry
	void * d_func_curry;
	cudaMalloc((void**)&d_func_curry, func_curry->size_in_byte);
	cudaMemcpy(d_func_curry, func_curry->ptr, func_curry->size_in_byte, cudaMemcpyHostToDevice);

	void * d_idx_func_curry;
	cudaMalloc((void**)&d_idx_func_curry, f_dst_pos_curry->size_in_byte);
	cudaMemcpy(d_idx_func_curry, f_dst_pos_curry->ptr, f_dst_pos_curry->size_in_byte, cudaMemcpyHostToDevice);

	// Run.
	const int n_elements =  dst->size_in_byte / sizeof(float);
	int blocksPerGrid = (n_elements/src_skip + 1 + threadsPerBlock - 1) / threadsPerBlock;
	_spmap<<<blocksPerGrid, threadsPerBlock>>>((float*) dst->ptr, (float *) src->ptr,
	  n_elements, src_skip, d_idx_myfunc, d_idx_func_curry, d_myfunc, d_func_curry);
	err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _spmap"  << "  ERROR " << err << std::endl;
	  assert(false);
	}
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	assert(err == cudaSuccess);

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

void GPUDriver::sapply(DeviceMemoryPointer * dst, FUNC_STRANSFORM * func, DeviceMemoryPointer * const func_curry){
	#ifdef _DO_ASSERT
	assert(dst->type==DEVICEMEMORY_LOCAL_RAM);
	assert(dst->size_in_byte % sizeof(float) == 0);
	#endif
	// TODO: Refactoring

	// First, create host version of func
	FUNC_STRANSFORM h_func;
	cudaMemcpyFromSymbol(&h_func, *func, sizeof(FUNC_STRANSFORM));
	FUNC_STRANSFORM d_myfunc = h_func;

	// Second, create a device version of func_curry
	void * d_func_curry;
	cudaMalloc((void**)&d_func_curry, func_curry->size_in_byte);
	cudaMemcpy(d_func_curry, func_curry->ptr, func_curry->size_in_byte, cudaMemcpyHostToDevice);

	// Run.
	const int n_elements =  dst->size_in_byte / sizeof(float);
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;
	_sapply<<<blocksPerGrid, threadsPerBlock>>>((float*) dst->ptr, n_elements, d_myfunc, d_func_curry);
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
  //cblas_sgemm(order, TA, TB, M, N, K, alpha,
  //  pA, LDA,
  //  pB, LDB,
  //  beta, pC, LDC);
}

void GPUDriver::selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1, 
DeviceMemoryPointer * src2, FUNC_SREDUCE * func, DeviceMemoryPointer * const func_curry){ 

	#ifdef _DO_ASSERT
	assert(dst->size_in_byte == src1->size_in_byte);
	assert(dst->size_in_byte == src2->size_in_byte);
	assert(dst->size_in_byte % sizeof(float) == 0);
	#endif
	// First, create host version of func
	FUNC_SREDUCE h_func;
	cudaMemcpyFromSymbol(&h_func, *func, sizeof(FUNC_SREDUCE));
	FUNC_SREDUCE d_myfunc = h_func;

	// Second, create a device version of func_curry
	void * d_func_curry;
	cudaMalloc((void**)&d_func_curry, func_curry->size_in_byte);
	cudaMemcpy(d_func_curry, func_curry->ptr, func_curry->size_in_byte, cudaMemcpyHostToDevice);

	// Run.
	const int n_elements =  dst->size_in_byte / sizeof(float);
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;
	_sreduce<<<blocksPerGrid, threadsPerBlock>>>((float*) dst->ptr, n_elements, 
	  (float*) src1->ptr, (float*) src2->ptr, d_myfunc, d_func_curry);
	err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _sreduce" << std::endl;
	  assert(false);
	}
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	assert(err == cudaSuccess);


}

FUNC_STRANSFORM * GPUDriver::srand_uni(float lower, float upper, DeviceMemoryPointer * arg){return NULL;}

FUNC_STRANSFORM * GPUDriver::srand_bern(float p, DeviceMemoryPointer * arg){return NULL;}

FUNC_STRANSFORM * GPUDriver::srand_gaussian(float mean, float std_dev, DeviceMemoryPointer * arg){return NULL;}

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
    sapply(arr, 
      (FUNC_STRANSFORM*)this->choose_ptr((void*)&__sconstant_initialize_helper,
                                          (void*)&_sconstant_initialize_helper),
      &pvalue);
}

void * GPUDriver::choose_ptr(void * host, void * device){
	return device;
}

template void GPUDriver::pmap2d_read_coalesce<_fpmap_id,_fmap_lower>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src, const struct PMapHelper args);


