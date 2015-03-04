
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "../src/sched/DeviceDriver.h"
#include "../src/sched/DeviceDriver_GPU.h"
#include "../src/sched/DeviceMemoryPointer.h"
#include <iostream>
#include <assert.h>
#include <functional>

__host__ __device__ float _f_add_one(float a, void * const arg){
  return a + *((float *) arg);
}
__device__ FUNC_STRANSFORM f_add_one = _f_add_one;

void test_array_equals_constant(float * array, int n_elements, float c){
	const float EPS = 0.01;
	for(int i=0;i<n_elements;i++){
		ASSERT_NEAR(array[i], c, EPS);
	}
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
	driver.sapply(&p_gpu, &f_add_one, &one);

  	cudaMemcpy(numbers, p_gpu.ptr, p_gpu.size_in_byte, cudaMemcpyDeviceToHost);
	test_array_equals_constant(numbers, 1000, 1.0);
}


/*

TEST(DeviceDriverTest, CPU_CONST_BERN) {
	float numbers[10000];
	test_array_set_constant(numbers, 10000, 1000.0);
	DeviceMemoryPointer_Local_RAM p(numbers, sizeof(float)*10000);
	CPUDriver driver;
	driver.sbernoulli_initialize(&p, 0.2);
	float sum = 0.0;
	for(int i=0;i<10000;i++){
		sum += numbers[i];
	}
	ASSERT_NEAR(sum/10000, 0.2, 0.1);
}

TEST(DeviceDriverTest, CPU_CONST_INIT) {
	float numbers[1000];
	test_array_set_constant(numbers, 1000, 1000.0);
	DeviceMemoryPointer_Local_RAM p(numbers, sizeof(float)*1000);
	CPUDriver driver;
	driver.sconstant_initialize(&p, 0.2);
	test_array_equals_constant(numbers, 1000, 0.2);
}

TEST(DeviceDriverTest, CPU_AXPY) {
	float numbers[1000];
	float numbers2[1000];
	test_array_set_constant(numbers, 1000, 1.0);
	test_array_set_constant(numbers2, 1000, 2.0);
	DeviceMemoryPointer_Local_RAM p1(numbers, sizeof(float)*1000);
	DeviceMemoryPointer_Local_RAM p2(numbers2, sizeof(float)*1000);
	CPUDriver driver;
	float alpha = 0.1;
	driver.smath_axpy(alpha, &p1, &p2);
	test_array_set_constant(numbers, 1000, 1.0);
	test_array_set_constant(numbers2, 1000, alpha*1.0+2.0);
}


TEST(DeviceDriverTest, CPU_AXPBY) {
	float numbers[1000];
	float numbers2[1000];
	test_array_set_constant(numbers, 1000, 1.0);
	test_array_set_constant(numbers2, 1000, 2.0);
	DeviceMemoryPointer_Local_RAM p1(numbers, sizeof(float)*1000);
	DeviceMemoryPointer_Local_RAM p2(numbers2, sizeof(float)*1000);
	CPUDriver driver;
	float alpha = 0.1;
	float beta = 0.5;
	driver.smath_axpby(alpha, &p1, beta, &p2);
	test_array_set_constant(numbers, 1000, 1.0);
	test_array_set_constant(numbers2, 1000, alpha*1.0+beta*2.0);
}
*/

//// MOVE THE FOLLOWING TO A SEPERATE TEST


/*


TEST(DeviceDriverTest, GPU_MEMCPY) {
	float numbers[1000];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p_gpu(0, NULL, sizeof(float)*1000);
	driver.malloc(&p_gpu);
	DeviceMemoryPointer_Local_GPURAM p_gpu2(0, NULL, sizeof(float)*1000);
	driver.malloc(&p_gpu2);
	driver.memset(p_gpu, 0);
	driver.memset(p_gpu2, 1);

	driver.memcpy(p_gpu2, p_gpu);
	clEnqueueReadBuffer (driver.queue, (cl_mem) p_gpu2.ptr, CL_TRUE, 0,
		sizeof (float) * 1000, numbers, 0, nullptr, nullptr);
	for(int i=0;i<1000;i++){
		EXPECT_EQ(numbers[i], 0.0);
	}
}

TEST(DeviceDriverTest, GPU_APPLY) {
	float numbers[1000];
	test_array_set_constant(numbers, 1000, 1000.0);
	DeviceMemoryPointer_Local_RAM p(numbers, sizeof(float)*1000);
	GPUDriver driver;
	auto f_set_to_zero = [](float & b) { b = 0; };
	driver.sapply(p, f_set_to_zero);
	test_array_equals_constant(numbers, 1000, 0.0);
}
*/


