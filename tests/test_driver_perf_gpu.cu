
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "../src/timer.h"
#include "../src/sched/DeviceDriver.h"
#include "../src/sched/DeviceDriver_GPU.h"
#include "../src/sched/DeviceMemoryPointer.h"
#include <iostream>
#include <assert.h>
#include <functional>
#include "../src/kernels/lowering.h"

void test_array_equals_constant(float * array, int n_elements, float c){
	const float EPS = 0.01;
	for(int i=0;i<n_elements;i++){
		ASSERT_NEAR(array[i], c, EPS);
	}
}

__host__ __device__ float _f_add_one(float a, void * const arg){
  return a + *((float *) arg);
}
__device__ FUNC_STRANSFORM f_add_one = _f_add_one;


__host__ __device__ float _f_set(float a, void * const arg){
  return *((float *) arg);
}
__device__ FUNC_STRANSFORM f_set = _f_set;


__host__ __device__ float _f_reduce(float a, float b, void * const arg){
	return a + b + *((float *) arg);
}
__device__ FUNC_SREDUCE f_reduce = _f_reduce;


__host__ __device__ size_t _f_idx_strid4_copy(size_t a, void * const arg){
	return a;
}
__device__ FUNC_IDX_MAPPING f_idx_strid4_copy = _f_idx_strid4_copy;

__host__ __device__ void _f_strid4_copy(void * dst, void * src, void * const arg){
	float * const _dst = (float *) dst;
	float * const _src = (float *) src;
	for(int i=0;i<4;i++){
		_dst[i] = _src[i] + *((float *) arg);
	}
}
__device__ FUNC_MM_MAPPING f_strid4_copy = _f_strid4_copy;


__global__ void vanilla_add_one(float * dst, int numElements){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < numElements){
  	dst[i] = dst[i] + 1;
  }
}


__global__ void vanilla_add_one_stride(float * dst, int numElements){
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * 8;
  #pragma unroll
  for(int j=0;j<8;j++){
	  if(i+j < numElements){
	    dst[i+j] = dst[i+j] + 1;
	  }
  }
}

__global__ void vanilla_copy(float * dst, float * src, int numElements){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  //if(i < numElements){
  //	dst[i] = dst[i] + 1;
  //}
  if(i < numElements){
	  dst[i] = src[i];
  }
}

__global__ void pmap_read_collapsed(float * data, float * output, const int iR, const int iC, const int iD, 
	const int iB, const int kR, const int kC, const int kD, const int kB){

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Calculate the workset of the current thread. 
	//
	const int ib = i/(iR*iC*iD);
	const int id = i/(iR*iC) % iD;
	const int ir = i/(iC) % (iR);
	const int ic = i% iC;
	
	const int o_base_col = ib * (iR-kR+1)*(iC-kC+1);
	const int o_base_row = id * kR * kC;
	const int oC = iB * (iR-kR+1)*(iC-kC+1);

	const float * const input_slice = &data[id * iR*iC + ib*iR*iC*iD];

	float input = input_slice[ir*iC+ic];
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

/*
__global__ void pmap_write_collapsed(float * data, float * output, const int oR, const int oC, const int oD, 
	const int oB, const int kR, const int kC, const int kD, const int kB){

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Calculate the workset of the current thread. 
	//
	const int oub = i/(oR*oC*oD);
	const int oud = i/(oR*oC) % oD;
	const int our = i/(oC) % (oR);
	const int ouc = i% oC;
	
	const int o_base_col = ib * (iR-kR+1)*(iC-kC+1);
	const int o_base_row = id * kR * kC;
	const int oC = iB * (iR-kR+1)*(iC-kC+1);

	const float * const input_slice = &data[id * iR*iC + ib*iR*iC*iD];

	float input = input_slice[ir*iC+ic];
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
*/

/**
struct PMapHelper{
  size_t dR, dC, dD, dB;  // dst RCDB
  size_t sR, sC, sD, sB;  // src RCDB
  size_t dBR, dBC;  // dst block
  size_t sBR, sBC;  // src block

  // lowering
  size_t kR, kC, kD, kB;  // kernel RCDB
};

struct 2DBlock{
  size_t r, c, d, b;
  size_t dr, dc;
};

struct PointIn2DBlock{
  float data;
  size_t r, c;
  2DBlock block;
};
**/

/**
typedef void (*FPMAP_ID) (2DBlock * const, const 2DBlock * const, const PMapHelper * const);
typedef void (*FPMAP_DATA_READC) (float * output_block, const 2DBlock * const, const * PointIn2DBlock const);
**/


__device__ void add_one(float * output){
	(*output) = (*output) + 1;
	printf("%f\n", (*output));
}

template <void (*T)(float *)>
__global__ void kernel2(float * output){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	T(&output[i]);
}

/*
TEST(DeviceDriverTest, GPU_APPLY_VANILLA) {

	// The goal is to see whether an exeuction without abstraction
	// is slow.

	int N = 10000;
	float * numbers = new float[N];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p_gpu(0, NULL, sizeof(float)*N);
	driver.malloc(&p_gpu);
	driver.memset(&p_gpu, 0);

	int threadsPerBlock = 256;
	const int n_elements =  N;
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;
	kernel2<add_one><<<blocksPerGrid, threadsPerBlock>>>((float*) p_gpu.ptr); // WARM UP
	cudaDeviceSynchronize();

	Timer t;
	kernel2<add_one><<<blocksPerGrid, threadsPerBlock>>>((float*) p_gpu.ptr);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _sapply" << "  ERROR " << err << std::endl;
	  assert(false);
	}
	double ti = t.elapsed();
	std::cout << "Time s    " << ti << std::endl;
}
*/

__device__
void _fpmap_id(Block2D * const output_block, const Block2D * const input_block, const PMapHelper * const args);

__device__
void _fmap_lower(float * output, const Block2D * const output_block, const PointIn2DBlock * const input_point, const PMapHelper * const args);


TEST(DeviceDriverTest, GPU_TEST_LOWERING_DRIVER) {
	const size_t iR = 55, iC = 55, iD = 48, iB = 256;
	const size_t kR = 5,  kC = 5,  kD = 96, kB = 128;
	const size_t oR = iR - kR + 1,
			  oC = iC - kC + 1,
			  oD = kB, oB = iB;

	std::cout << "T " << oR*oC*iB*iD*kR*kC << std::endl;

	GPUDriver driver;

	DeviceMemoryPointer_Local_GPURAM p_data(0, NULL, sizeof(float)*iR*iC*iD*iB);
	DeviceMemoryPointer_Local_GPURAM p_kernel(0, NULL, sizeof(float)*kR*kC*kD*kB);
	DeviceMemoryPointer_Local_GPURAM p_output(0, NULL, sizeof(float)*oR*oC*iD*iB*kR*kC);
	driver.malloc(&p_data); driver.malloc(&p_kernel); driver.malloc(&p_output);
	driver.memset(&p_data, 0); driver.memset(&p_kernel, 0); driver.memset(&p_output, 0);

	PMapHelper args;
	args.dR = kR*kC*iD; args.dC = oR*oC*iB; args.dD = 1; args.dB = 1;
	args.sR = iR; args.sC = iC; args.sD = iD; args.sB = iB;
	args.dBR = args.dR; args.dBC = args.dC;
	args.sBR = 32; args.sBC = 32;
	args.kR = kR; args.kC = kC; args.kD = kD; args.kB = kB;

	//driver.pmap2d_read_coalesce(&p_output, &p_data, args);

	driver.pmap2d_read_coalesce<_fpmap_id,_fmap_lower>(&p_output, &p_data, args);

	Timer t;
	driver.pmap2d_read_coalesce<_fpmap_id,_fmap_lower>(&p_output, &p_data, args);
	
	//driver.pmap2d_read_coalesce(&p_output, &p_data, args);
	

	double ti = t.elapsed();
	double read_byte = iR*iB*iC*iD*sizeof(float);
	double write_byte = sizeof(float)*oR*oC*iD*iB*kR*kC;
	double bandwidth_mbs = 1.0*(read_byte+write_byte)/1024/1024/ti;
	std::cout << "Time s    " << ti << std::endl;
	std::cout << "Band MB/s " << bandwidth_mbs << std::endl;

}


/*
TEST(DeviceDriverTest, GPU_TEST_LOWERING1) {

	const size_t iR = 55, iC = 55, iD = 48, iB = 256;
	const size_t kR = 5,  kC = 5,  kD = 96, kB = 128;
	const size_t oR = iR - kR + 1,
			  oC = iC - kC + 1,
			  oD = kB, oB = iB;

	std::cout << "T " << oR*oC*iB*iD*kR*kC << std::endl;

	GPUDriver driver;

	DeviceMemoryPointer_Local_GPURAM p_data(0, NULL, sizeof(float)*iR*iC*iD*iB);
	DeviceMemoryPointer_Local_GPURAM p_kernel(0, NULL, sizeof(float)*kR*kC*kD*kB);
	DeviceMemoryPointer_Local_GPURAM p_output(0, NULL, sizeof(float)*oR*oC*iD*iB*kR*kC);
	driver.malloc(&p_data); driver.malloc(&p_kernel); driver.malloc(&p_output);
	driver.memset(&p_data, 0); driver.memset(&p_kernel, 0); driver.memset(&p_output, 0);

	pmap_read_collapsed<<<1, 1>>>((float*) p_data.ptr,
		(float *) p_output.ptr, iR, iC, iD, iB, kR, kC, kD, kB);

	int threadsPerBlock = 256;
	const int n_elements = iR*iC*iD*iB;
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;
	Timer t;
	std::cout << "START KERNEL " << std::endl;
	//pmap_read_collapsed<<<blocksPerGrid, threadsPerBlock>>>((float*) p_data.ptr,
	//	(float *) p_output.ptr, iR, iC, iD, iB, kR, kC, kD, kB);
	pmap_write_collapsed<<<blocksPerGrid, threadsPerBlock>>>((float*) p_data.ptr,
		(float *) p_output.ptr, kR*kC*iD, oR*oC*iB, 1, 1, kR, kC, kD, kB);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _sapply" << "  ERROR " << err << std::endl;
	  assert(false);
	}

	double ti = t.elapsed();
	double read_byte = iR*iB*iC*iD*sizeof(float);
	double write_byte = sizeof(float)*oR*oC*iD*iB*kR*kC;
	//double flop = N;
	double bandwidth_mbs = 1.0*(read_byte+write_byte)/1024/1024/ti;
	//double flop_g = 1.0*(flop)/1024/1024/ti;
	std::cout << "Time s    " << ti << std::endl;
	//std::cout << "In. size  " << 1.0*read_byte/1024/1024 << std::endl;
	std::cout << "Band MB/s " << bandwidth_mbs << std::endl;
	//std::cout << "FLOP G/s  " << flop_g << std::endl;
}
*/

/*
TEST(DeviceDriverTest, GPU_APPLY_VANILLA) {

	// The goal is to see whether an exeuction without abstraction
	// is slow.

	int N = 400000000;
	float * numbers = new float[N];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p_gpu(0, NULL, sizeof(float)*N);
	driver.malloc(&p_gpu);
	driver.memset(&p_gpu, 0);

	DeviceMemoryPointer_Local_GPURAM p_gpu2(0, NULL, sizeof(float)*N);
	driver.malloc(&p_gpu2);
	driver.memset(&p_gpu2, 0);

	int threadsPerBlock = 256;
	const int n_elements =  N;
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;
	vanilla_copy<<<blocksPerGrid, threadsPerBlock>>>((float*) p_gpu.ptr, (float*) p_gpu2.ptr, n_elements); // WARM UP
	cudaDeviceSynchronize();

	Timer t;
	vanilla_copy<<<blocksPerGrid, threadsPerBlock>>>((float*) p_gpu.ptr, (float*) p_gpu2.ptr, n_elements);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _sapply" << "  ERROR " << err << std::endl;
	  assert(false);
	}
	double ti = t.elapsed();
	double read_byte = N*sizeof(float);
	double write_byte = N*sizeof(float);
	double flop = N;
	double bandwidth_mbs = 1.0*(read_byte+write_byte)/1024/1024/ti;
	double flop_g = 1.0*(flop)/1024/1024/ti;
	std::cout << "Time s    " << ti << std::endl;
	std::cout << "In. size  " << 1.0*read_byte/1024/1024 << std::endl;
	std::cout << "Band MB/s " << bandwidth_mbs << std::endl;
	std::cout << "FLOP G/s  " << flop_g << std::endl;
}
*/



/*
TEST(DeviceDriverTest, GPU_APPLY) {

	// The goal is to see whether a single APPLY of the GPU kernel is slow

	int N = 1000000000;
	float * numbers = new float[N];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p_gpu(0, NULL, sizeof(float)*N);
	driver.malloc(&p_gpu);
	driver.memset(&p_gpu, 0);

	float one = 1.0;
	DeviceMemoryPointer_Local_RAM p_one(&one, sizeof(float));

	Timer t;
	driver.sapply(&p_gpu, &f_add_one, &p_one);
	cudaDeviceSynchronize();

	double ti = t.elapsed();
	double read_byte = N*sizeof(float);
	double write_byte = N*sizeof(float);
	double flop = N;
	double bandwidth_mbs = 1.0*(read_byte+write_byte)/1024/1024/ti;
	double flop_g = 1.0*(flop)/1024/1024/ti;
	std::cout << "Time s    " << ti << std::endl;
	std::cout << "In. size  " << 1.0*read_byte/1024/1024 << std::endl;
	std::cout << "Band MB/s " << bandwidth_mbs << std::endl;
	std::cout << "FLOP G/s  " << flop_g << std::endl;

  	cudaMemcpy(numbers, p_gpu.ptr, p_gpu.size_in_byte, cudaMemcpyDeviceToHost);
	test_array_equals_constant(numbers, N, 1.0);
}
*/

/*
TEST(DeviceDriverTest, GPU_APPLY_VANILLA) {

	// The goal is to see whether an exeuction without abstraction
	// is slow.

	int N = 1000000000;
	float * numbers = new float[N];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p_gpu(0, NULL, sizeof(float)*N);
	driver.malloc(&p_gpu);
	driver.memset(&p_gpu, 0);

	int threadsPerBlock = 256;
	const int n_elements =  N;
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;
	Timer t;
	vanilla_add_one<<<blocksPerGrid, threadsPerBlock>>>((float*) p_gpu.ptr, n_elements);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _sapply" << "  ERROR " << err << std::endl;
	  assert(false);
	}
	double ti = t.elapsed();
	double read_byte = N*sizeof(float);
	double write_byte = N*sizeof(float);
	double flop = N;
	double bandwidth_mbs = 1.0*(read_byte+write_byte)/1024/1024/ti;
	double flop_g = 1.0*(flop)/1024/1024/ti;
	std::cout << "Time s    " << ti << std::endl;
	std::cout << "In. size  " << 1.0*read_byte/1024/1024 << std::endl;
	std::cout << "Band MB/s " << bandwidth_mbs << std::endl;
	std::cout << "FLOP G/s  " << flop_g << std::endl;
}
*/

/*
TEST(DeviceDriverTest, GPU_APPLY_VANILLA_STRIDE) {

	// The goal is to see whether an exeuction without abstraction
	// is slow.

	int N = 1000000000;
	float * numbers = new float[N];
	int stride = 8;

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p_gpu(0, NULL, sizeof(float)*N);
	driver.malloc(&p_gpu);
	driver.memset(&p_gpu, 0);

	int threadsPerBlock = 256;
	const int n_elements =  N;
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock / stride;
	Timer t;
	vanilla_add_one_stride<<<blocksPerGrid, threadsPerBlock>>>((float*) p_gpu.ptr, n_elements);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
	  std::cout << "Fail to launch _sapply" << "  ERROR " << err << std::endl;
	  assert(false);
	}
	double ti = t.elapsed();
	double read_byte = N*sizeof(float);
	double write_byte = N*sizeof(float);
	double flop = N;
	double bandwidth_mbs = 1.0*(read_byte+write_byte)/1024/1024/ti;
	double flop_g = 1.0*(flop)/1024/1024/ti;
	std::cout << "Time s    " << ti << std::endl;
	std::cout << "In. size  " << 1.0*read_byte/1024/1024 << std::endl;
	std::cout << "Band MB/s " << bandwidth_mbs << std::endl;
	std::cout << "FLOP G/s  " << flop_g << std::endl;

	// As epected, with stride it is slower.
}
*/

/*
TEST(DeviceDriverTest, GPU_PMAP) {
	const int N=1000000;
	float numbers[1000000];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p1(0, NULL, sizeof(float)*1000000);
	DeviceMemoryPointer_Local_GPURAM p2(0, NULL, sizeof(float)*1000000);
	driver.malloc(&p1); driver.malloc(&p2);

	float one = 1.0;
	DeviceMemoryPointer_Local_RAM p_one(&one, sizeof(float));

	driver.sconstant_initialize(&p1, 0.2);
	driver.sconstant_initialize(&p2, 3.0);

	driver.parallel_map(&p2, &p1, 4, &f_idx_strid4_copy, &p_one, &f_strid4_copy, &p_one);
  	cudaMemcpy(numbers, p2.ptr, p2.size_in_byte, cudaMemcpyDeviceToHost);
	
	test_array_equals_constant(numbers, 1000000, 1.2);
}
*/

/*
TEST(DeviceDriverTest, GPU_REDUCE) {
	float numbers[1000];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p1(0, NULL, sizeof(float)*1000);
	DeviceMemoryPointer_Local_GPURAM p2(0, NULL, sizeof(float)*1000);
	DeviceMemoryPointer_Local_GPURAM p3(0, NULL, sizeof(float)*1000);
	driver.malloc(&p1); driver.malloc(&p2); driver.malloc(&p3);

	float one = 1.0;
	DeviceMemoryPointer_Local_RAM p_one(&one, sizeof(float));

	driver.sconstant_initialize(&p1, 1.0);
	driver.sconstant_initialize(&p2, 2.0);
	driver.sconstant_initialize(&p3, 3.0);

	driver.selementwise_reduce2(&p3, &p1, &p2, &f_reduce, &p_one);
  	cudaMemcpy(numbers, p3.ptr, p3.size_in_byte, cudaMemcpyDeviceToHost);
	
	test_array_equals_constant(numbers, 1000, 4.0);
}


TEST(DeviceDriverTest, GPU_MEMSET) {
	
	float numbers[1000];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p_gpu(0, NULL, sizeof(float)*1000);
	driver.malloc(&p_gpu);
	driver.memset(&p_gpu, 1);

  	cudaMemcpy(numbers, p_gpu.ptr, p_gpu.size_in_byte, cudaMemcpyDeviceToHost);
	for(int i=0;i<1000;i++){EXPECT_EQ(numbers[i] != 0.0, true);}

	driver.memset(&p_gpu, 0);
  	cudaMemcpy(numbers, p_gpu.ptr, p_gpu.size_in_byte, cudaMemcpyDeviceToHost);
	test_array_equals_constant(numbers, 1000, 0.0);

	driver.free(&p_gpu);
}

TEST(DeviceDriverTest, GPU_MEMCPY) {
	float numbers[1000];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p_gpu(0, NULL, sizeof(float)*1000);
	driver.malloc(&p_gpu);
	DeviceMemoryPointer_Local_GPURAM p_gpu2(0, NULL, sizeof(float)*1000);
	driver.malloc(&p_gpu2);
	driver.memset(&p_gpu, 0);
	driver.memset(&p_gpu2, 1);

	driver.memcpy(&p_gpu2, &p_gpu);
	driver.memset(&p_gpu, 0);
  	cudaMemcpy(numbers, p_gpu2.ptr, p_gpu.size_in_byte, cudaMemcpyDeviceToHost);
	test_array_equals_constant(numbers, 1000, 0.0);
}


TEST(DeviceDriverTest, GPU_APPLY) {

	float numbers[1000];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p_gpu(0, NULL, sizeof(float)*1000);
	driver.malloc(&p_gpu);
	driver.memset(&p_gpu, 0);

	float one = 1.0;
	DeviceMemoryPointer_Local_RAM p_one(&one, sizeof(float));
	driver.sapply(&p_gpu, &f_add_one, &p_one);

  	cudaMemcpy(numbers, p_gpu.ptr, p_gpu.size_in_byte, cudaMemcpyDeviceToHost);
	test_array_equals_constant(numbers, 1000, 1.0);
}

TEST(DeviceDriverTest, GPU_AXPY) {

	float numbers[1000];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p1(0, NULL, sizeof(float)*1000);
	DeviceMemoryPointer_Local_GPURAM p2(0, NULL, sizeof(float)*1000);
	driver.malloc(&p1); driver.malloc(&p2);

	float one = 1.0;
	DeviceMemoryPointer_Local_RAM p_one(&one, sizeof(float));
	driver.sapply(&p1, &f_set, &p_one);

	float two = 2.0;
	DeviceMemoryPointer_Local_RAM p_two(&two, sizeof(float));
	driver.sapply(&p2, &f_set, &p_two);

	float alpha = 0.1;
	driver.smath_axpy(alpha, &p1, &p2);
  	cudaMemcpy(numbers, p2.ptr, p2.size_in_byte, cudaMemcpyDeviceToHost);
	test_array_equals_constant(numbers, 1000, 2.1);

}

TEST(DeviceDriverTest, GPU_AXPBY) {
	float numbers[1000];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p1(0, NULL, sizeof(float)*1000);
	DeviceMemoryPointer_Local_GPURAM p2(0, NULL, sizeof(float)*1000);
	driver.malloc(&p1); driver.malloc(&p2);

	float one = 1.0;
	DeviceMemoryPointer_Local_RAM p_one(&one, sizeof(float));
	driver.sapply(&p1, &f_set, &p_one);

	float two = 2.0;
	DeviceMemoryPointer_Local_RAM p_two(&two, sizeof(float));
	driver.sapply(&p2, &f_set, &p_two);

	float alpha = 0.1;
	float beta = 0.5;
	driver.smath_axpby(alpha, &p1, beta, &p2);
  	cudaMemcpy(numbers, p2.ptr, p2.size_in_byte, cudaMemcpyDeviceToHost);

	test_array_equals_constant(numbers, 1000, 1.1);
}

TEST(DeviceDriverTest, GPU_CONST_BERN) {
	float numbers[10000];
	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p(0, NULL, sizeof(float)*10000);
	driver.malloc(&p);

	float one = 10000.0;
	DeviceMemoryPointer_Local_RAM p_one(&one, sizeof(float));
	driver.sapply(&p, &f_set, &p_one);

	driver.sbernoulli_initialize(&p, 0.2);

  	cudaMemcpy(numbers, p.ptr, p.size_in_byte, cudaMemcpyDeviceToHost);

	float sum = 0.0;
	for(int i=0;i<10000;i++){
		sum += numbers[i];
	}
	ASSERT_NEAR(sum/10000, 0.2, 0.1);
}


TEST(DeviceDriverTest, GPU_CONST_INIT) {
	float numbers[10000];
	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p(0, NULL, sizeof(float)*10000);
	driver.malloc(&p);

	float one = 10000.0;
	DeviceMemoryPointer_Local_RAM p_one(&one, sizeof(float));
	driver.sapply(&p, &f_set, &p_one);

	driver.sconstant_initialize(&p, 0.2);

  	cudaMemcpy(numbers, p.ptr, p.size_in_byte, cudaMemcpyDeviceToHost);
	test_array_equals_constant(numbers, 1000, 0.2);
}
*/


