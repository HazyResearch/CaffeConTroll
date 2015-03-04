
#include "gtest/gtest.h"
#include "glog/logging.h"
#include "../src/sched/DeviceDriver.h"
#include "../src/sched/DeviceDriver_CPU.h"
#include "../src/sched/DeviceMemoryPointer.h"
#include <iostream>
#include <assert.h>
#include <functional>

void test_array_equals_constant(float * array, int n_elements, float c){
	const float EPS = 0.01;
	for(int i=0;i<n_elements;i++){
		ASSERT_NEAR(array[i], c, EPS);
	}
}

void test_array_set_constant(float * array, int n_elements, float c){
	for(int i=0;i<n_elements;i++){
		array[i] = c;
	}
	test_array_equals_constant(array, n_elements, c);
}

TEST(DeviceDriverTest, CPU_MEMSET) {
	float numbers[1000];
	test_array_set_constant(numbers, 1000, 1000.0);
	DeviceMemoryPointer_Local_RAM p(numbers, sizeof(float)*1000);
	CPUDriver driver;
	driver.memset(p, 0);
	test_array_equals_constant(numbers, 1000, 0.0);
}

TEST(DeviceDriverTest, CPU_MEMCPY) {
	float numbers[1000];
	float numbers2[1000];
	test_array_set_constant(numbers, 1000, 1000.0);
	test_array_set_constant(numbers2, 1000, 0.0);
	DeviceMemoryPointer_Local_RAM p1(numbers, sizeof(float)*1000);
	DeviceMemoryPointer_Local_RAM p2(numbers2, sizeof(float)*1000);
	CPUDriver driver;
	driver.memcpy(p2, p1);
	test_array_equals_constant(numbers, 1000, 1000.0);
	test_array_equals_constant(numbers2, 1000, 1000.0);
}

TEST(DeviceDriverTest, CPU_APPLY) {
	float numbers[1000];
	test_array_set_constant(numbers, 1000, 1000.0);
	DeviceMemoryPointer_Local_RAM p(numbers, sizeof(float)*1000);
	CPUDriver driver;
	auto f_set_to_zero = [](float & b) { b = 0; };
	driver.sapply(p, 1000, f_set_to_zero);
	test_array_equals_constant(numbers, 1000, 0.0);
}


TEST(DeviceDriverTest, CPU_CONST_INIT) {
	float numbers[1000];
	test_array_set_constant(numbers, 1000, 1000.0);
	DeviceMemoryPointer_Local_RAM p(numbers, sizeof(float)*1000);
	CPUDriver driver;
	driver.constant_initialize(p, 0.2, 1000);
	test_array_equals_constant(numbers, 1000, 0.2);
}

TEST(DeviceDriverTest, CPU_CONST_BERN) {
	float numbers[10000];
	test_array_set_constant(numbers, 1000, 1000.0);
	DeviceMemoryPointer_Local_RAM p(numbers, sizeof(float)*1000);
	CPUDriver driver;
	driver.bernoulli_initialize(p, 10000, 0.2);
	float sum = 0.0;
	for(int i=0;i<10000;i++){sum += numbers[i];}
	ASSERT_NEAR(sum/10000, 0.2, 0.1);
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
	driver.smath_axpy(alpha, p1, p2);
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
	driver.smath_axpby(alpha, p1, beta, p2);
	test_array_set_constant(numbers, 1000, 1.0);
	test_array_set_constant(numbers2, 1000, alpha*1.0+beta*2.0);
}




