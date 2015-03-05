
#include "DeviceDriver.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

#include "DeviceDriver_GPU.cuh"

#ifndef _DEVICE_DRIVER_GPU_H
#define _DEVICE_DRIVER_GPU_H

class GPUDriver : public DeviceDriver{
public:

  int gpu_id = 0;

  int threadsPerBlock = 256;

  cublasStatus_t status;
  
  cublasHandle_t handle;

  cudaError_t err;

  GPUDriver(){
    cublasCreate(&handle);

  }

  DeviceMemoryPointer * get_device_pointer(void * ptr, size_t size_in_byte){
    // TODO: This has memory leak! Refactor it!
    return new DeviceMemoryPointer_Local_GPURAM(gpu_id, ptr, size_in_byte);
  }

  virtual void malloc(DeviceMemoryPointer * dst){
    cudaMalloc((void**)&dst->ptr, dst->size_in_byte);
  }

  virtual void free(DeviceMemoryPointer * dst){
    cudaFree(dst->ptr);
  }

  void memcpy(DeviceMemoryPointer * dst, DeviceMemoryPointer * src){
#ifdef _DO_ASSERT
    assert(dst->type==DEVICEMEMORY_LOCAL_RAM);
    assert(src->type==DEVICEMEMORY_LOCAL_RAM);
    assert(dst->size_in_byte == src->size_in_byte);
#endif
    cudaMemcpy(dst->ptr, src->ptr, dst->size_in_byte, cudaMemcpyDeviceToDevice);
  }

  void memset(DeviceMemoryPointer * dst, const char value){
#ifdef _DO_ASSERT
    assert(dst->type==DEVICEMEMORY_LOCAL_RAM);
#endif
    cudaMemset(dst->ptr, value, dst->size_in_byte);
  }

  void parallel_map(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
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
      std::cout << "Fail to launch _sapply" << std::endl;
      assert(false);
    }

    cudaFree(d_func_curry);
    cudaFree(d_idx_func_curry);

  }

    void smath_axpy(const float alpha, DeviceMemoryPointer * X, DeviceMemoryPointer * Y)  { 
#ifdef _DO_ASSERT
    assert(X->type==DEVICEMEMORY_LOCAL_RAM);
    assert(Y->type==DEVICEMEMORY_LOCAL_RAM);
    assert(X->size_in_byte==Y->size_in_byte);
#endif
      int n_elements = X->size_in_byte / sizeof(float);
      status = cublasSaxpy(handle, n_elements, &alpha, (float*)X->ptr, 1, (float*)Y->ptr, 1);
      assert(status == CUBLAS_STATUS_SUCCESS);
    }

  void sapply(DeviceMemoryPointer * dst, FUNC_STRANSFORM * func, DeviceMemoryPointer * const func_curry){
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
      std::cout << "Fail to launch _sapply" << std::endl;
      assert(false);
    }

    cudaFree(d_func_curry);
  }

    void smath_axpby(const float alpha, DeviceMemoryPointer * X, const float beta, DeviceMemoryPointer * Y) { 
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

    void set_num_threads(const int nThreads) { 
    }


  void sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
        int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
        float beta, float * pC, int LDC){
      //cblas_sgemm(order, TA, TB, M, N, K, alpha,
      //  pA, LDA,
      //  pB, LDB,
      //  beta, pC, LDC);
  }

  void selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1, 
    DeviceMemoryPointer * src2, FUNC_SREDUCE * func, DeviceMemoryPointer * const func_curry){ 
      // This lambda should be easier for compiler to inline than a function pointer
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

  }

  FUNC_STRANSFORM * srand_uni(float lower, float upper, DeviceMemoryPointer * arg){return NULL;}

  FUNC_STRANSFORM * srand_bern(float p, DeviceMemoryPointer * arg){return NULL;}

  FUNC_STRANSFORM * srand_gaussian(float mean, float std_dev, DeviceMemoryPointer * arg){return NULL;}

  /**
   * This function is called only once. So its speed does not matter.
   * TODO: Wrap this up with CURAND.
   **/
  void sinitialize_xavier(DeviceMemoryPointer *arr, const size_t n_batch) {
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
  void sbernoulli_initialize(DeviceMemoryPointer *arr, const float p) {
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
  void sgaussian_initialize(DeviceMemoryPointer *arr, const float mean, const float std_dev) {
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
  
  void * choose_ptr(void * host, void * device){
    return device;
  }

  using DeviceDriver::sconstant_initialize;

  using DeviceDriver::smath_apply_grad;

private:

  random_device rd;
};

#endif

