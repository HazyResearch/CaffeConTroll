
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "../src/sched/DeviceDriver.h"
#include "../src/sched/DeviceDriver_GPU.h"
#include "../src/sched/DeviceDriver_GPU.cuh"
#include "../src/sched/DeviceMemoryPointer.h"
#include <iostream>
#include <assert.h>
#include <functional>

#include "../src/kernels/lowering.h"

inline void test_array_equals_constant(float * array, int n_elements, float c){
	const float EPS = 0.01;
	for(int i=0;i<n_elements;i++){
		ASSERT_NEAR(array[i], c, EPS);
	}
}

TEST(DeviceDriverTest, GPU_PMAP) {
	float numbers[1000];

	GPUDriver driver;
	DeviceMemoryPointer_Local_GPURAM p1(0, NULL, sizeof(float)*1000);
	DeviceMemoryPointer_Local_GPURAM p2(0, NULL, sizeof(float)*1000);
	driver.malloc(&p1); driver.malloc(&p2);

	float one = 1.0;
	DeviceMemoryPointer_Local_RAM p_one(&one, sizeof(float));

	driver.sconstant_initialize(&p1, 0.2);
	driver.sconstant_initialize(&p2, 3.0);

	driver.parallel_map<_f_idx_strid4_copy,_f_strid4_copy>(&p2, &p1, 4*sizeof(float), &p_one, &p_one);
  	cudaMemcpy(numbers, p2.ptr, p2.size_in_byte, cudaMemcpyDeviceToHost);
	
	test_array_equals_constant(numbers, 1000, 1.2);
}

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

	driver.selementwise_reduce2<_f_reduce>(&p3, &p1, &p2, &p_one);
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
	driver.sapply<_f_add_one>(&p_gpu, &p_one);

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
	driver.sapply<_f_set>(&p1, &p_one);

	float two = 2.0;
	DeviceMemoryPointer_Local_RAM p_two(&two, sizeof(float));
	driver.sapply<_f_set>(&p2, &p_two);

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
	driver.sapply<_f_set>(&p1, &p_one);

	float two = 2.0;
	DeviceMemoryPointer_Local_RAM p_two(&two, sizeof(float));
	driver.sapply<_f_set>(&p2, &p_two);

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
	driver.sapply<_f_set>(&p, &p_one);

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
	driver.sapply<_f_set>(&p, &p_one);

	driver.sconstant_initialize(&p, 0.2);

  	cudaMemcpy(numbers, p.ptr, p.size_in_byte, cudaMemcpyDeviceToHost);
	test_array_equals_constant(numbers, 1000, 0.2);
}



