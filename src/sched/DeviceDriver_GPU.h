
#include "DeviceDriver.h"

#ifndef _DEVICE_DRIVER_GPU_H
#define _DEVICE_DRIVER_GPU_H

class GPUDriver : public DeviceDriver{
public:

  DeviceMemoryPointer * get_device_pointer(void * ptr, size_t size_in_byte){
    // TODO: This has memory leak! Refactor it!
  	return new DEVICEMEMORY_LOCAL_GPURAM(ptr, size_in_byte);
  }

  virtual void malloc(DeviceMemoryPointer * dst){
    //dst->ptr = ::malloc(dst->size_in_byte);
  }

  virtual void free(DeviceMemoryPointer * dst){
    //::free(dst->ptr);
  }

  void memcpy(DeviceMemoryPointer dst, DeviceMemoryPointer src){
#ifdef _DO_ASSERT
    assert(dst.type==DEVICEMEMORY_LOCAL_GPURAM);
    assert(src.type==DEVICEMEMORY_LOCAL_GPURAM);
    assert(dst.size_in_byte == src.size_in_byte);
#endif
    //char *s1 = (char*) dst.ptr;
    //const char *s2 = (const char*) src.ptr;
    //size_t n = dst.size_in_byte;
    //for(; 0<n; --n)*s1++ = *s2++;
  }

  void memset(DeviceMemoryPointer dst, const char value){
#ifdef _DO_ASSERT
    assert(dst.type==DEVICEMEMORY_LOCAL_GPURAM);
#endif
    //char *s1 = (char*) dst.ptr;
    //size_t n = dst.size_in_byte;
    //for(; 0<n; --n)*s1++ = value;
  }

  void parallel_map(DeviceMemoryPointer dst, DeviceMemoryPointer src, 
    size_t src_skip, std::function<size_t(size_t)> f_dst_pos,
    std::function<void(void *, void *)> func){

  	//char * p_dst = (char*) dst.ptr;
  	//char * p_src = (char*) src.ptr;
  	//const size_t src_size = src.size_in_byte;
  	//for(size_t i=0; i<src_size; i+=src_skip){
  	//	func(&p_dst[f_dst_pos(i)], &p_src[i]);
  	//}

  }

  void smath_axpy(const float alpha, DeviceMemoryPointer X, DeviceMemoryPointer Y)  { 
#ifdef _DO_ASSERT
    assert(X.type==DEVICEMEMORY_LOCAL_RAM);
    assert(Y.type==DEVICEMEMORY_LOCAL_RAM);
    assert(X.size_in_byte==Y.size_in_byte);
#endif
    //cblas_saxpy(X.size_in_byte/sizeof(float), alpha, (float *) X.ptr, 1, (float *) Y.ptr, 1); 
  }

  void sapply(DeviceMemoryPointer dst, std::function<void(float&)> func){
#ifdef _DO_ASSERT
    assert(dst.type==DEVICEMEMORY_LOCAL_RAM);
    assert(dst.size_in_byte % sizeof(float) == 0);
#endif
    //const size_t n_element = dst.size_in_byte/sizeof(float);
    //float * p = (float*) dst.ptr;
    //for(size_t i=0;i<n_element;i++){
    //  func(*(p++));
    //}
  }

	void smath_axpby(const float alpha, DeviceMemoryPointer X, const float beta, DeviceMemoryPointer Y) { 
#ifdef _DO_ASSERT
	  assert(X.size_in_byte == Y.size_in_byte);
		assert(X.size_in_byte % sizeof(float) == 0);
#endif
  	//cblas_saxpby(X.size_in_byte/sizeof(float), alpha, (float*)X.ptr, 1, beta, (float*) Y.ptr, 1); 
  }

  void set_num_threads(const int nThreads) { 
    //openblas_set_num_threads(nThreads); 
  }

  void sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
        int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
        float beta, float * pC, int LDC){

    //cblas_sgemm(order, TA, TB, M, N, K, alpha,
    //  pA, LDA,
    //  pB, LDB,
    //  beta, pC, LDC);
  }

  void selementwise_reduce2(DeviceMemoryPointer dst, DeviceMemoryPointer src1, 
    DeviceMemoryPointer src2, std::function<float(float,float)> FUNC){ 
      // This lambda should be easier for compiler to inline than a function pointer
#ifdef _DO_ASSERT
    assert(dst.size_in_byte == src1.size_in_byte);
    assert(dst.size_in_byte == src2.size_in_byte);
    assert(dst.size_in_byte % sizeof(float) == 0);
#endif
    //const size_t n_element = dst.size_in_byte / sizeof(float);
    //float * const p_dst = (float*) dst.ptr;
    //const float * const p_src1 = (float*) src1.ptr;
    //const float * const p_src2 = (float*) src2.ptr; 
    //for(size_t i = 0; i < n_element; i++){
    //  p_dst[i] = FUNC(p_src1[i], p_src2[i]);
    //}
  }

  std::function<void(float&)> srand_uni(float lower, float upper){
    //mt19937 *gen = new mt19937(rd());
    //uniform_real_distribution<float> *uni = new 
    //  uniform_real_distribution<float>(lower, upper);
    //return [=](float & b) { b = (*uni)(*gen); };
  }

  std::function<void(float&)> srand_bern(float p){
    //mt19937 *gen = new mt19937(rd());
    //bernoulli_distribution * bern = new bernoulli_distribution(p);
    //return [=](float & b) { b = (*bern)(*gen); };
  }

  std::function<void(float&)> srand_gaussian(float mean, float std_dev){
    //mt19937 *gen = new mt19937(rd());
    //normal_distribution<float> *gaussian = new
    //  normal_distribution<float>(mean, std_dev);
    //return [=](float & b) { b = (*gaussian)(*gen); };
  }

  using DeviceDriver::sinitialize_xavier;

  using DeviceDriver::sbernoulli_initialize;

  using DeviceDriver::sgaussian_initialize;

  using DeviceDriver::sconstant_initialize;

  using DeviceDriver::smath_apply_grad;

private:

  random_device rd;
};

#endif

