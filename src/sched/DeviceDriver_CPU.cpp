
#include "DeviceDriver_CPU.h"

CPUDriver::CPUDriver(){

}

DeviceMemoryPointer * CPUDriver::get_device_pointer(void * ptr, size_t size_in_byte){
	assert(false);
}

void CPUDriver::malloc(DeviceMemoryPointer * dst){
	dst->ptr = ::malloc(dst->size_in_byte);
}

void CPUDriver::free(DeviceMemoryPointer * dst){
    ::free(dst->ptr);
}

void CPUDriver::memcpy(DeviceMemoryPointer * dst, DeviceMemoryPointer * src){
    char *s1 = (char*) dst->ptr;
    const char *s2 = (const char*) src->ptr;
    size_t n = dst->size_in_byte;
    for(; 0<n; --n)*s1++ = *s2++;
}

void CPUDriver::memset(DeviceMemoryPointer * dst, const char value){
    char *s1 = (char*) dst->ptr;
    size_t n = dst->size_in_byte;
    for(; 0<n; --n)*s1++ = value;
}

template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
void CPUDriver::pmap2d_read_coalesce(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
const struct PMapHelper args){

}

template<FUNC_IDX_MAPPING f_dst_pos, FUNC_MM_MAPPING func>
void CPUDriver::parallel_map(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
DeviceMemoryPointer * const func_curry){
  	char * p_dst = (char*) dst->ptr;
  	char * p_src = (char*) src->ptr;
  	const size_t src_size = src->size_in_byte;
  	for(size_t i=0; i<src_size; i+=src_skip){
  		func(&p_dst[f_dst_pos(i, f_dst_pos_curry->ptr)], &p_src[i], func_curry->ptr);
  	}
}

void CPUDriver::smath_axpy(const float alpha, DeviceMemoryPointer * X, DeviceMemoryPointer * Y){
	cblas_saxpy(X->size_in_byte/sizeof(float), alpha, (float *) X->ptr, 1, (float *) Y->ptr, 1); 
}

void CPUDriver::smath_axpby(const float alpha, DeviceMemoryPointer * X, const float beta, DeviceMemoryPointer * Y){
#ifdef _USE_OPENBLAS
    cblas_saxpby(X->size_in_byte/sizeof(float), alpha, (float*)X->ptr, 1, beta, (float*) Y->ptr, 1); 
#elif _USE_ATLAS
    catlas_saxpby(X->size_in_byte/sizeof(float), alpha, (float*)X->ptr, 1, beta, (float*) Y->ptr, 1); 
#elif _VANILLA_BLAS
    #warning "[PERFORMANCE WARNING] Using hand-written BLAS calls. Hope you have a good compiler!"
	const int N = X->size_in_byte/sizeof(float);
	float * _X = X->ptr;
	float * _Y = Y->ptr;
	for(int i = N; i > 0; _X++, _Y++, --i) {
		*Y = alpha**_X + beta* *_Y;
  	}
#else
	#error "Select a BLAS framework."
#endif
}

void CPUDriver::set_num_threads(const int nThreads){
#ifdef _USE_OPENBLAS
    openblas_set_num_threads(nThreads);
#elif _USE_ATLAS
    set_num_threads(nThreads);
#elif _VANILLA_BLAS

#else
	#error "Select a BLAS framework."
#endif
}


void CPUDriver::sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
    int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
    float beta, float * pC, int LDC){

	cblas_sgemm(order, TA, TB, M, N, K, alpha,
		pA, LDA,
		pB, LDB,
		beta, pC, LDC);

}

template<FUNC_STRANSFORM func>
void CPUDriver::sapply(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry){
    const size_t n_element = dst->size_in_byte/sizeof(float);
    float * p = (float*) dst->ptr;
    for(size_t i=0;i<n_element;i++){
      *(p) = func(*(p), func_curry->ptr);
      p++;
    }
}

template<FUNC_SREDUCE func>
void CPUDriver::selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1, 
DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry){
    const size_t n_element = dst->size_in_byte / sizeof(float);
    float * const p_dst = (float*) dst->ptr;
    const float * const p_src1 = (float*) src1->ptr;
    const float * const p_src2 = (float*) src2->ptr; 
    for(size_t i = 0; i < n_element; i++){
      p_dst[i] = func(p_src1[i], p_src2[i], func_curry->ptr);
    }
}

void CPUDriver::sinitialize_xavier(DeviceMemoryPointer *arr, const size_t n_batch){
	const size_t n_arr_elements = arr->size_in_byte / sizeof(float);
	const size_t fan_in = n_arr_elements / n_batch;
	const float scale = sqrt(3.0 / fan_in);

	mt19937 gen(rd());
	uniform_real_distribution<float> uni(-scale, scale);
	float * temp = (float*) arr->ptr;
	for(int i=0;i<n_arr_elements;i++){
	  temp[i] = uni(gen);
	}
}

void CPUDriver::sbernoulli_initialize(DeviceMemoryPointer *arr, const float p){
	const size_t n_arr_elements = arr->size_in_byte / sizeof(float);

	mt19937 gen(rd());
	bernoulli_distribution bern(p);
	float * temp = (float*) arr->ptr;
	for(int i=0;i<n_arr_elements;i++){
	  temp[i] = bern(gen);
	}
}

void CPUDriver::sgaussian_initialize(DeviceMemoryPointer *arr, const float mean, const float std_dev){
	const size_t n_arr_elements = arr->size_in_byte / sizeof(float);
	mt19937 gen(rd());
	normal_distribution<float> gaussian(mean, std_dev);
	float * temp = (float*) arr->ptr;
	for(int i=0;i<n_arr_elements;i++){
	  temp[i] = gaussian(gen);
	}
}

void CPUDriver::sconstant_initialize(DeviceMemoryPointer *arr, const float value){
	const size_t n_arr_elements = arr->size_in_byte / sizeof(float);
	float * const temp = (float*) arr->ptr;
	for(int i=0;i<n_arr_elements;i++){
	  temp[i] = value;
	}
}

void * CPUDriver::choose_ptr(void * host, void * device){
	return host;
}

/**
 * This is necessary for template to be instantiated.
 */
template void CPUDriver::pmap2d_read_coalesce<_fpmap_id,_fmap_lower>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src, const struct PMapHelper args);

template void CPUDriver::pmap2d_read_coalesce<_fpmap_id,_fmap_remap>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src, const struct PMapHelper args);


template void CPUDriver::parallel_map<_f_idx_strid4_copy,_f_strid4_copy>
	(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, size_t src_skip, 
		DeviceMemoryPointer * const f_dst_pos_curry, DeviceMemoryPointer * const func_curry);

template void CPUDriver::sapply<_f_add_one>(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry);

template void CPUDriver::sapply<_f_set>(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry);

template void CPUDriver::selementwise_reduce2<_f_reduce>(DeviceMemoryPointer * dst, 
	DeviceMemoryPointer * src1, DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);


