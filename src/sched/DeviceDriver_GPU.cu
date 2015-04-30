
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

// NOTE: Ensure there are no race conditions when calling this function
// See backward bias calculation for conv and fullyconnected - need to call
// parallel_map multiple times because otherwise output indices overlap
template<FUNC_IDX_MAPPING idx_func, FUNC_MM_MAPPING func>
__global__ void _spmap(float * dst, float * src, int numElements, int srcSkip,
  void * const idx_func_curry, void * const func_curry){
  char * p_dst = (char*) dst;
  char * p_src = (char*) src;
  const size_t src_size = numElements*srcSkip;
  size_t i = (blockDim.x * blockIdx.x + threadIdx.x) * srcSkip;
  if(i < src_size){
    func(&p_dst[idx_func(i, idx_func_curry)], &p_src[i], func_curry, idx_func(i, idx_func_curry));
  }
}

__global__ void _parallel_lower_cube(float * dst, float * src, const struct PMapHelper args){

  // Read arguments
  const int iD = args.sD;
  const int iR = args.sR;
  const int iC = args.sC;
  const int kR = args.kR;
  const int kC = args.kC;
  const int iB = args.sB;
  const int p  = args.padding;
  const int s  = args.stride;
  const int oR = (iR + 2*p - kR) / s + 1;
  const int oC = (iC + 2*p - kC) / s + 1; 
  
  // Get the right loop element
  const int iB_idx   = blockIdx.x;
  const int iD_idx   = blockIdx.y;
  const int oRoC_idx = blockIdx.z;
  const int oR_idx   = oRoC_idx/oC;
  const int oC_idx   = oRoC_idx%oC;
  const int kR_idx   = threadIdx.x;
  const int kC_idx   = threadIdx.y;
  
  const int out_r = iB_idx*oR*oC + oR_idx*oC + oC_idx;
  const int out_c = iD_idx*kR*kC + kR_idx*kC + kC_idx;

  if ( (oR_idx*s-p+kR_idx) >= 0 && (oR_idx*s-p+kR_idx) < iR && (oC_idx*s-p+kC_idx) >= 0 && (oC_idx*s-p+kC_idx) < iC ) {
    dst[out_r*iD*kR*kC + out_c] = src[iB_idx*iC*iR*iD + iD_idx*iR*iC + (oR_idx*s-p+kR_idx)*iC + (oC_idx*s-p+kC_idx)];
  } else {
    dst[out_r*iD*kR*kC + out_c] = 0;
  }
}

__global__ void _parallel_inverse_lower_cube(float * dst, float * src, const struct _inverse_lower_cube_arg_helper args){

  // Read arguments
  const int oC = args.data_output_width;
  const int oR = args.data_output_height;
  const int k = args.kernel_size;
  const int s = args.stride;
  const int p = args.padding;
  const int iR = args.iR;
  const int iC = args.iC;
  const int iD = args.iD;
  const unsigned int iB = args.iB;

  // Get the right loop element
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  // SHADJIS TODO: These / and % not needed if using multi-dimensional blocks
  const int b =   (i / (iC*iR*iD));
  const int tmp = (i % (iC*iR*iD));
  const int c =  tmp / (iC * iR);
  const int h = (tmp / iC) % iR + p;
  const int w =  tmp % iC + p;
  
  const int w_col_start = (w < k) ? 0 : (w - k) / s + 1;
  const int w_col_end = device_min(w / s + 1, oC);
  const int h_col_start = (h < k) ? 0 : (h - k) / s + 1;
  const int h_col_end = device_min(h / s + 1, oR);
  
  // SHADJIS TODO: Not sure why but the way we store batches is
  // different from Caffe so this part had to be changed. Probably
  // something to do with the old/unnecessary CcT decision to flip
  // the gemm order everywhere. So instead of storing each batch
  // of src one after the other we interleave them. I think this is
  // pretty stupid but I don't feel like rewriting everything now.
  // Probably leave that for when we do fusion.
  const int offset = (c*k*k*iB + h*k*iB + w*iB + b)*oR*oC;
  const int coeff_h_col = (1 - s*k*oR*iB)*oC;
  const int coeff_w_col = (1 - s*oR*oC*iB);
  float sum = 0;
  for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
    for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
      sum += src[offset + h_col * coeff_h_col + w_col * coeff_w_col];
    }
  }
  dst[i] = sum;
}

template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
__global__ void _spmap_readc(float* dst, float * src, PMapHelper args){
	const size_t block_x = blockIdx.x;
	const size_t block_y = blockIdx.y;

	const size_t nCblock = (args.sC + args.sBC-1)/args.sBC;

	Block2D input_block;
	input_block.r = block_x / nCblock;
	input_block.c = block_x % nCblock;
	input_block.d = block_y % args.sD;
	input_block.b = block_y / args.sD;
	input_block.dr = args.sR;
	input_block.dc = args.sC;

	Block2D output_block;
	f_id(&output_block, &input_block, &args);

	const size_t datar = threadIdx.y + input_block.r * args.sBR;
	const size_t datac = threadIdx.x + input_block.c * args.sBC;

	PointIn2DBlock point;
	point.block = input_block;
    
    const size_t src_idx = args.sR * args.sC * (args.sD * input_block.b + input_block.d) +
            datar * args.sC +
            datac;

    // Check if in bounds
    if (datar < args.sR && datac < args.sC)
    {
        point.data = src[src_idx];
        point.r = datar;
        point.c = datac;
        f_data(dst, &output_block, &point, &args);
    }
}

template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
void GPUDriver::lower_cube(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct PMapHelper args){
    pmap2d_read_coalesce<f_id, f_data>(dst, src, args);
    // lower_cube_helper(dst, src, args);
}

// SHADJIS TODO: This is a more parallel forward lowering which matches the implementation
// on the CPU. But because it flips the lowering (transposes) also need to change cuBLAS
// flags below.
void GPUDriver::lower_cube_helper(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct PMapHelper args){
    
    const int iD = args.sD;
    const int kr = args.kR;
    const int kc = args.kC;
    const int iB = args.sB;
    const int p  = args.padding;
    const int s  = args.stride;
    const int iR = args.sR;
    const int iC = args.sC;
    const int oR = (iR + 2*p - kr) / s + 1;
    const int oC = (iC + 2*p - kc) / s + 1; 
    
    dim3 numBlocks(iB, iD, oR*oC);
    // SHADJIS TODO: Should fix the number of threads (e.g. 256, 1024) since now
    // warps are under-utilized for small k
    dim3 threadsPerBlock(kr, kc);
    // SHADJIS TODO: Call something like _spmap_readc instead
    // SHADJIS TODO: Add a check here for too many blocks like sapply, or make multi-dimensional like _spmap_readc
    cudaGetLastError(); // Reset the error status to success
    _parallel_lower_cube<<<numBlocks, threadsPerBlock>>>((float*) dst->ptr, (float *) src->ptr, args);
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "Fail to launch _parallel_lower_cube"  << "  ERROR " << err << std::endl;
      assert(false);
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "Fail to sync _parallel_lower_cube"  << "  ERROR " << err << std::endl;
      assert(false);
    }
}


// Note: lower_cube and also inverse_lower_cube are special-case functions, i.e.
// they do not use parallel map + kernel callbacks. They could use that interface
// but it may be easier for fusion to keep them separate.
void GPUDriver::inverse_lower_cube(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _inverse_lower_cube_arg_helper args){

    const int iD = args.iD;
    const int iR = args.iR;
    const int iC = args.iC;
    const unsigned int iB = args.iB;
    const int num_parallel_elements = iR*iC*iD*iB;
    int blocksPerGrid = (num_parallel_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    // SHADJIS TODO: Call something like _spmap_readc instead
    // SHADJIS TODO: Add a check here for too many blocks like sapply, or make multi-dimensional like _spmap_readc
    cudaGetLastError(); // Reset the error status to success
    _parallel_inverse_lower_cube<<<blocksPerGrid, threadsPerBlock>>>((float*) dst->ptr, (float *) src->ptr, args);
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "Fail to launch _parallel_inverse_lower_cube"  << "  ERROR " << err << std::endl;
      assert(false);
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "Fail to sync _parallel_inverse_lower_cube"  << "  ERROR " << err << std::endl;
      assert(false);
    }
}


// SHADJIS TODO: Why is the interface for this is different from parallel_map?
// Here we pass in the args directly whereas parallel_map gets pointers to
// the args already allocated on the device
template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
void GPUDriver::pmap2d_read_coalesce(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct PMapHelper args){

	// input block sizes
	size_t sBR = args.sBR, sBC = args.sBC;
    
	dim3 threadsPerBlock(sBC, sBR);	// trivial impl -- each input pixel is a single thread
	// The number of blocks and threads are chosen to map to each pixel in input (1 thread/pixel)
	dim3 numBlocks(((args.sR + sBR-1)/sBR)*((args.sC + sBC-1)/sBC), args.sD*args.sB);

	cudaGetLastError(); // Reset the error status to success
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
	cudaGetLastError(); // Reset the error status to success
	const int n_elements =  src->size_in_byte / src_skip;
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;
	// SHADJIS TODO: Why call _spmap and not _spmap_readc?
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

void GPUDriver::math_saxpy(const float alpha, DeviceMemoryPointer * X, DeviceMemoryPointer * Y) const { 
#ifdef _DO_ASSERT
	assert(X->type==DEVICEMEMORY_LOCAL_RAM);
	assert(Y->type==DEVICEMEMORY_LOCAL_RAM);
	assert(X->size_in_byte==Y->size_in_byte);
#endif
  int n_elements = X->size_in_byte / sizeof(float);
  cublasStatus_t status = cublasSaxpy(handle, n_elements, &alpha, (float*)X->ptr, 1, (float*)Y->ptr, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);
}

void GPUDriver::math_saxpy(const int nElements, const float alpha, float * X, float * Y) const { 
  cublasStatus_t status = cublasSaxpy(handle, nElements, &alpha, X, 1, Y, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);
}

template<FUNC_STRANSFORM func>
void GPUDriver::sapply(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry){
	#ifdef _DO_ASSERT
	assert(dst->type==DEVICEMEMORY_LOCAL_GPURAM);
	assert(dst->size_in_byte % sizeof(float) == 0);
	#endif
	// TODO: Refactoring

	// Second, create a device version of func_curry
	void * d_func_curry;
	cudaMalloc((void**)&d_func_curry, func_curry->size_in_byte);
	cudaMemcpy(d_func_curry, func_curry->ptr, func_curry->size_in_byte, cudaMemcpyHostToDevice);

	cudaGetLastError(); // Reset the error status to success
	int n_elements =  dst->size_in_byte / sizeof(float);
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;

	// Check if we are trying to initialize something huge. In that case call multiple times.
	// SHADJIS TODO: Could make multi-dimensional like _spmap_readc instead
	const int num_calls = (blocksPerGrid + max_cuda_blocks - 1) / max_cuda_blocks;
	if (num_calls > 1) {
		blocksPerGrid = max_cuda_blocks;
	}
	for (int call_counter=0; call_counter < num_calls; ++call_counter)
	{
		_sapply<func><<<blocksPerGrid, threadsPerBlock>>>((float*) (dst->ptr + call_counter*blocksPerGrid*threadsPerBlock), n_elements, d_func_curry);
		err = cudaGetLastError();
		if(err != cudaSuccess){
			std::cout << "Fail to launch _sapply" << "  ERROR " << err << std::endl;
			assert(false);
		}
		n_elements -= blocksPerGrid*threadsPerBlock; // Decrement #elements left to process
	}
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	assert(err == cudaSuccess);

	cudaFree(d_func_curry);
}

void GPUDriver::math_saxpby(const float alpha, DeviceMemoryPointer * X, const float beta, DeviceMemoryPointer * Y) const { 
#ifdef _DO_ASSERT
  assert(X->size_in_byte == Y->size_in_byte);
  assert(X->size_in_byte % sizeof(float) == 0);
#endif

  int n_elements = X->size_in_byte / sizeof(float);
  cublasStatus_t status = cublasSscal(handle, n_elements, &beta, (float*)Y->ptr, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);

  status = cublasSaxpy(handle, n_elements, &alpha, (float*)X->ptr, 1, (float*)Y->ptr, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);

}

void GPUDriver::math_saxpby(const int nElements, const float alpha, float * X, const float beta, float * Y) const { 
  cublasStatus_t status = cublasSscal(handle, nElements, &beta, Y, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);

  status = cublasSaxpy(handle, nElements, &alpha, X, 1, Y, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);

}

void GPUDriver::set_num_threads(const int nThreads) { 
}


void GPUDriver::sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
    int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
    float beta, float * pC, int LDC){
  
	// SHADJIS TODO: See comment in Kernel.h regarding transpose. For the CPU it is fastest 
	// to lower like equation 4 of "Formulation of Type 1 Lowering with Padding and Stride"
	// but the GPU currently lowers as the transpose of what the CPU does. For now I change
	// the parameters in here to match. It's pretty complicated to get these cuBLAS parameters
	// right because cuBLAS also assumes things are stored in column-major order. It's made
	// more complicated because the lowering on CPU and GPU differs (by transpose), so making
	// the lowered versions match would make this easier to follow.

	if(TA == CblasNoTrans && TB == CblasNoTrans){

		cublasOperation_t ta = CUBLAS_OP_N;
		// tb should also be no trans, but is transposed to match cpu lowering
		cublasOperation_t tb = CUBLAS_OP_T; 

		// cublas expects col major, so we change the parameters accordingly
		status = cublasSgemm(handle, tb, ta, N, M, K, &alpha, 
			pB, K, pA, K, &beta, pC, N); 

		cudaDeviceSynchronize();
		err = cudaGetLastError();
		assert(err == cudaSuccess);

		assert(status == CUBLAS_STATUS_SUCCESS);

	}else if(TA == CblasTrans && TB == CblasNoTrans){

		cublasOperation_t ta = CUBLAS_OP_T;
		cublasOperation_t tb = CUBLAS_OP_N;

		// cublas expects col major, so we change the parameters accordingly
		status = cublasSgemm(handle, tb, ta, N, M, K, &alpha, 
			pB, N, pA, M, &beta, pC, N); 

		cudaDeviceSynchronize();
		err = cudaGetLastError();
		assert(err == cudaSuccess);

		assert(status == CUBLAS_STATUS_SUCCESS);

	}else if(TA == CblasNoTrans && TB == CblasTrans){

		cublasOperation_t ta = CUBLAS_OP_N;
		// tb should be trans, but is transposed to match cpu lowering
		cublasOperation_t tb = CUBLAS_OP_N;

		// cublas expects col major, so we change the parameters accordingly
		status = cublasSgemm(handle, tb, ta, N, M, K, &alpha, 
			pB, N, pA, K, &beta, pC, N); 

		cudaDeviceSynchronize();
		err = cudaGetLastError();
		assert(err == cudaSuccess);

		assert(status == CUBLAS_STATUS_SUCCESS);

	}else if(TA == CblasTrans && TB == CblasTrans){

		cublasOperation_t ta = CUBLAS_OP_T;
		cublasOperation_t tb = CUBLAS_OP_T;

		// cublas expects col major, so we change the parameters accordingly
		status = cublasSgemm(handle, tb, ta, N, M, K, &alpha, 
			pB, K, pA, M, &beta, pC, N); 

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

// SHADJIS TODO: No need to template this if we switch to new lowering
template void GPUDriver::lower_cube<_fpmap_id,_fmap_lower>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src, const struct PMapHelper args);

template void GPUDriver::pmap2d_read_coalesce<_fpmap_id,_fmap_remap>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src, const struct PMapHelper args);

/** All template instantiations for parallel_map **/
template void GPUDriver::parallel_map<_f_idx_strid4_copy,_f_strid4_copy>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// inverse_lower_cube
template void GPUDriver::parallel_map<_f_src_to_dst_inverse_lower_cube,_f_inverse_lower_cube>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Conv/FC Bias forward
template void GPUDriver::parallel_map<_f_src_to_dst_bias_forward,_f_bias_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Conv/FC Bias backward
template void GPUDriver::parallel_map<_f_src_to_dst_bias_backward,_f_bias_backward>(DeviceMemoryPointer * dst,
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

