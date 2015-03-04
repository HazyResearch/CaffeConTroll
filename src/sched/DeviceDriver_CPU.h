
#include "DeviceDriver.h"

#ifndef _DEVICE_DRIVER_CPU_H
#define _DEVICE_DRIVER_CPU_H

class CPUDriver : public DeviceDriver{
public:

  DeviceMemoryPointer * get_device_pointer(void * ptr, size_t size_in_byte){
    // TODO: This has memory leak! Refactor it!
  	return new DeviceMemoryPointer_Local_RAM(ptr, size_in_byte);
  }

  virtual void malloc(DeviceMemoryPointer * dst){
    dst->ptr = ::malloc(dst->size_in_byte);
  }

  virtual void free(DeviceMemoryPointer * dst){
    ::free(dst->ptr);
  }

  void memcpy(DeviceMemoryPointer dst, DeviceMemoryPointer src){
#ifdef _DO_ASSERT
    assert(dst.type==DEVICEMEMORY_LOCAL_RAM);
    assert(src.type==DEVICEMEMORY_LOCAL_RAM);
    assert(dst.size_in_byte == src.size_in_byte);
#endif
    char *s1 = (char*) dst.ptr;
    const char *s2 = (const char*) src.ptr;
    size_t n = dst.size_in_byte;
    for(; 0<n; --n)*s1++ = *s2++;
  }

  void memset(DeviceMemoryPointer dst, const char value){
#ifdef _DO_ASSERT
    assert(dst.type==DEVICEMEMORY_LOCAL_RAM);
#endif
    char *s1 = (char*) dst.ptr;
    size_t n = dst.size_in_byte;
    for(; 0<n; --n)*s1++ = value;
  }

  void parallel_map(DeviceMemoryPointer dst, DeviceMemoryPointer src, 
    size_t src_skip, FUNC_IDX_MAPPING f_dst_pos, void * const f_dst_pos_curry,
    FUNC_MM_MAPPING func, void * const func_curry){

  	char * p_dst = (char*) dst.ptr;
  	char * p_src = (char*) src.ptr;
  	const size_t src_size = src.size_in_byte;
  	for(size_t i=0; i<src_size; i+=src_skip){
  		(*func)(&p_dst[(*f_dst_pos)(i, f_dst_pos_curry)], &p_src[i], func_curry);
  	}

  }

    void smath_axpy(const float alpha, DeviceMemoryPointer X, DeviceMemoryPointer Y)  { 
#ifdef _DO_ASSERT
    assert(X.type==DEVICEMEMORY_LOCAL_RAM);
    assert(Y.type==DEVICEMEMORY_LOCAL_RAM);
    assert(X.size_in_byte==Y.size_in_byte);
#endif
      cblas_saxpy(X.size_in_byte/sizeof(float), alpha, (float *) X.ptr, 1, (float *) Y.ptr, 1); 
    }

  void sapply(DeviceMemoryPointer dst, FUNC_STRANSFORM func, void * const func_curry){
#ifdef _DO_ASSERT
    assert(dst.type==DEVICEMEMORY_LOCAL_RAM);
    assert(dst.size_in_byte % sizeof(float) == 0);
#endif
    const size_t n_element = dst.size_in_byte/sizeof(float);
    float * p = (float*) dst.ptr;
    for(size_t i=0;i<n_element;i++){
      *(p) = (*func)(*(p), func_curry);
      p++;
    }
  }

#ifdef _USE_OPENBLAS
  	void smath_axpby(const float alpha, DeviceMemoryPointer X, const float beta, DeviceMemoryPointer Y) { 
#ifdef _DO_ASSERT
  		assert(X.size_in_byte == Y.size_in_byte);
  		assert(X.size_in_byte % sizeof(float) == 0);
#endif
    	cblas_saxpby(X.size_in_byte/sizeof(float), alpha, (float*)X.ptr, 1, beta, (float*) Y.ptr, 1); 
    }
    void set_num_threads(const int nThreads) { openblas_set_num_threads(nThreads); }
#elif _USE_ATLAS
  	void smath_axpby(const float alpha, DeviceMemoryPointer X, const float beta, DeviceMemoryPointer Y) { 
#ifdef _DO_ASSERT
  		assert(X.size_in_byte == Y.size_in_byte);
  		assert(X.size_in_byte % sizeof(float) == 0);
#endif
    	catlas_saxpby(X.size_in_byte/sizeof(float), alpha, (float*)X.ptr, 1, beta, (float*) Y.ptr, 1); 
    }
    void set_num_threads(const int nThreads) {  set_num_threads(nThreads); }
#elif _VANILLA_BLAS
    #warning "[PERFORMANCE WARNING] Using hand-written BLAS calls. Hope you have a good compiler!"
  	void smath_axpby(const float alpha, DeviceMemoryPointer X, const float beta, DeviceMemoryPointer Y) { 
#ifdef _DO_ASSERT
  		assert(X.size_in_byte == Y.size_in_byte);
  		assert(X.size_in_byte % sizeof(float) == 0);
#endif
    	const int N = X.size_in_byte/sizeof(float);
    	float * _X = X.ptr;
    	float * _Y = Y.ptr;
    	for(int i = N; i > 0; _X++, _Y++, --i) {
			*Y = alpha**_X + beta* *_Y;
      	}
    }
    void set_num_threads(const int nThreads) {   }
#else
	#error "Select a BLAS framework."
#endif

  void sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
        int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
        float beta, float * pC, int LDC){

      cblas_sgemm(order, TA, TB, M, N, K, alpha,
        pA, LDA,
        pB, LDB,
        beta, pC, LDC);

  }

  void selementwise_reduce2(DeviceMemoryPointer dst, DeviceMemoryPointer src1, 
    DeviceMemoryPointer src2, FUNC_SREDUCE func, void * const func_curry){ 
      // This lambda should be easier for compiler to inline than a function pointer
#ifdef _DO_ASSERT
    assert(dst.size_in_byte == src1.size_in_byte);
    assert(dst.size_in_byte == src2.size_in_byte);
    assert(dst.size_in_byte % sizeof(float) == 0);
#endif
    const size_t n_element = dst.size_in_byte / sizeof(float);
    float * const p_dst = (float*) dst.ptr;
    const float * const p_src1 = (float*) src1.ptr;
    const float * const p_src2 = (float*) src2.ptr; 
    for(size_t i = 0; i < n_element; i++){
      p_dst[i] = (*func)(p_src1[i], p_src2[i], func_curry);
    }
  }


  class _srand_uni_arg_helper{
  public:
    random_device rd;
    mt19937 gen;
    uniform_real_distribution<float> uni;
    _srand_uni_arg_helper(float lower, float upper):
      gen(mt19937(rd())), uni(uniform_real_distribution<float>(lower, upper)){}
  };
  static float _srand_uni_helper(float a, void * const arg){
    _srand_uni_arg_helper * const p_arg = 
      reinterpret_cast<_srand_uni_arg_helper* const>(arg);
    return p_arg->uni(p_arg->gen);
  }
  FUNC_STRANSFORM srand_uni(float lower, float upper, void ** arg){
    *arg = new _srand_uni_arg_helper(lower, upper);
    return _srand_uni_helper;
  }


  class _srand_bern_arg_helper{
  public:
    random_device rd;
    mt19937 gen;
    bernoulli_distribution bern;
    _srand_bern_arg_helper(float p):
      gen(mt19937(rd())), bern(bernoulli_distribution(p)){}
  };
  static float _srand_bern_helper(float a, void * const arg){
    _srand_bern_arg_helper * const p_arg = (_srand_bern_arg_helper*) arg;
    return p_arg->bern(p_arg->gen);
  }
  FUNC_STRANSFORM srand_bern(float p, void ** arg){
    *arg = new _srand_bern_arg_helper(p);
    return _srand_bern_helper;
  }


  class _srand_gaussian_arg_helper{
  public:
    random_device rd;
    mt19937 gen;
    normal_distribution<float> gaussian;
    _srand_gaussian_arg_helper(float mean, float std_dev):
      gen(mt19937(rd())), gaussian(normal_distribution<float>(mean, std_dev)){}
  };
  static float _srand_gaussian_helper(float a, void * const arg){
    _srand_gaussian_arg_helper * const p_arg = (_srand_gaussian_arg_helper*) arg;
    return p_arg->gaussian(p_arg->gen);
  }
  FUNC_STRANSFORM srand_gaussian(float mean, float std_dev, void ** arg){
    *arg = new _srand_gaussian_arg_helper(mean, std_dev);
    return _srand_gaussian_helper;
  }

  using DeviceDriver::sbernoulli_initialize;

  using DeviceDriver::sinitialize_xavier;

  using DeviceDriver::sgaussian_initialize;
  
  using DeviceDriver::sconstant_initialize;

  using DeviceDriver::smath_apply_grad;

private:

  random_device rd;
};

#endif

