
#include "DeviceDriver_CPU.h"
#include "../kernels/include.hxx"

CPUDriver::CPUDriver() {

}

CPUDriver::~CPUDriver() {

}

DeviceMemoryPointer * CPUDriver::get_device_pointer(void * ptr, size_t size_in_byte) {
  return new DeviceMemoryPointer_Local_RAM(ptr, size_in_byte);
}

void CPUDriver::malloc(DeviceMemoryPointer * dst) {
  dst->ptr = ::malloc(dst->size_in_byte);
}

void CPUDriver::free(DeviceMemoryPointer * dst) {
  ::free(dst->ptr);
}

void CPUDriver::memcpy(DeviceMemoryPointer * dst, DeviceMemoryPointer * src) {
  char *s1 = (char*) dst->ptr;
  const char *s2 = (const char*) src->ptr;
  size_t n = dst->size_in_byte;
  for (; 0<n; --n)*s1++ = *s2++;
}

void CPUDriver::memset(DeviceMemoryPointer * dst, const char value) {
  char *s1 = (char*) dst->ptr;
  size_t n = dst->size_in_byte;
  for (; 0<n; --n)*s1++ = value;
}

template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
// SHADJIS TODO: This function uses float, not char like parallel_map
inline void _spmap_cpu(float* const dst, float * const src, PMapHelper args,
    const size_t block_x, const size_t block_y, const size_t thread_x, const size_t thread_y) {

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
    
    const size_t datar = thread_y    + input_block.r * args.sBR;
    const size_t datac = thread_x    + input_block.c * args.sBC;

    PointIn2DBlock point;
    point.block = input_block;

    const size_t src_idx = args.sR * args.sC * (args.sD * input_block.b + input_block.d) + 
            datar * args.sC + 
            datac;

    // Check if in bounds
    if (datar < args.sR && datac < args.sC)
    {
#ifdef _DO_ASSERT
        assert(args.sR * args.sC * args.sD * args.sB > src_idx);
#endif
        point.data = src[src_idx];
        point.r = datar;
        point.c = datac;
        f_data(dst, &output_block, &point, &args);
    }
}

// SHADJIS TODO: This function can be made faster by blocking, etc.
// SHADJIS TODO: For now rather than use pmap2d_read_coalesce for lowering
// I wrote a separate type 1 lowering (lower_cube) which is faster due to
// better access patterns. However, alternatively we can continue to call
// pmap2d_read_coalesce but make this function faster, e.g. by blocking
// the loops.
template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
inline void CPUDriver::pmap2d_read_coalesce(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const struct PMapHelper args) {

    // input block sizes
    size_t sBR = args.sBR, sBC = args.sBC;
    assert(sBC*sBR > 0);
    size_t n_thread_per_block_C = sBC;
    size_t n_thread_per_block_R = sBR;
    size_t n_block_X = ((args.sR + sBR-1)/sBR)*((args.sC + sBC-1)/sBC);
    size_t n_block_Y = args.sD*args.sB;

    for (size_t block_x = 0; block_x < n_block_X; ++block_x) {
        for (size_t block_y = 0; block_y < n_block_Y; ++block_y) {
            for (size_t thread_x = 0; thread_x < n_thread_per_block_C; ++thread_x) {
                for (size_t thread_y = 0; thread_y < n_thread_per_block_R; ++thread_y) {
                    _spmap_cpu<f_id,f_data>((float*) dst->ptr, (float*) src->ptr, args,
                    block_x, block_y, thread_x, thread_y);
                }
            }
        }
    }
}

#define FW_LOWER_OPTIMIZATIONS 1

// Type 1 lowering as defined in the paper
// Note the result D_hat is transposed compared to the expected format
template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
inline void CPUDriver::lower_cube(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const struct PMapHelper args) {

    DeviceMemoryPointer * const device_mem_ptr_D = src;
    DeviceMemoryPointer * const device_mem_ptr_D_lowered = dst;
    const int n = args.sR;
    const int d = args.sD;
    const int k = args.kR;
    const int s = args.stride;
    const int p = args.padding;
    const int b = args.sB;

#ifdef _DO_ASSERT
    assert(args.sR == args.sC);
    assert(args.kR == args.kC);
#endif

    // Implement equation 4 of
    // "Formulation of Type 1 Lowering with Padding and Stride"
    // Optimizations are possible, e.g. lifting out padding checks and blocking loops
    const float * const D = (float *)device_mem_ptr_D->ptr;
    float * const D_lowered = (float *)device_mem_ptr_D_lowered->ptr;
    const int m = ( (n + 2*p - k) / s + 1 );

#if FW_LOWER_OPTIMIZATIONS
    const int D_BLOCK_SIZE = 8;
    
    // Handle blocking of d

    // Special case:  If d is a multiple of block size no need for internal checking
    // Currently commented out, need to do more experiments
    if (d % D_BLOCK_SIZE == 0)
    {
        for (int bi=0; bi<b; ++bi) {
            for (int r=0; r<m; ++r) {
                for (int c=0; c<m; ++c) {
                    float *const current_row = &(D_lowered[(bi*m*m + r*m + c)*k*k*d]);
                    for (int Dd_block=0; Dd_block<d; Dd_block+=D_BLOCK_SIZE) {
                        for (int Dr=0; Dr<k; ++Dr) {
                            for (int Dc=0; Dc<k; ++Dc) {
                                for (int Dd=Dd_block; Dd<Dd_block+D_BLOCK_SIZE; ++Dd) {
                                    if ( (r*s-p+Dr) >= 0 && (r*s-p+Dr) < n && (c*s-p+Dc) >= 0 && (c*s-p+Dc) < n ) {
                                        current_row[Dd*k*k + Dr*k + Dc] = D[bi*n*n*d + Dd*n*n + (r*s-p+Dr)*n + (c*s-p+Dc)];
                                    } else {
                                        current_row[Dd*k*k + Dr*k + Dc] = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Special case: If d < D_BLOCK_SIZE no need to block it at all
    // Currently commented out, need to do more experiments
    else if (d < D_BLOCK_SIZE)
    {
        for (int bi=0; bi<b; ++bi) {
            for (int r=0; r<m; ++r) {
                for (int c=0; c<m; ++c) {
                    float *const current_row = &(D_lowered[(bi*m*m + r*m + c)*k*k*d]);
                    for (int Dd=0; Dd<d; ++Dd) {
                        for (int Dr=0; Dr<k; ++Dr) {
                            for (int Dc=0; Dc<k; ++Dc) {
                                if ( (r*s-p+Dr) >= 0 && (r*s-p+Dr) < n && (c*s-p+Dc) >= 0 && (c*s-p+Dc) < n ) {
                                    current_row[Dd*k*k + Dr*k + Dc] = D[bi*n*n*d + Dd*n*n + (r*s-p+Dr)*n + (c*s-p+Dc)];
                                } else {
                                    current_row[Dd*k*k + Dr*k + Dc] = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Most general case
    else
    {
        for (int bi=0; bi<b; ++bi) {
            for (int r=0; r<m; ++r) {
                //const int start_Dr = std::max(0,p-r*s);
                //const int end_Dr =   std::min(k,n-r*s+p);
                for (int c=0; c<m; ++c) {
                    //const int start_Dc = std::max(0,p-c*s);
                    //const int end_Dc =   std::min(k,n-c*s+p);
                    float *const current_row = &(D_lowered[(bi*m*m + r*m + c)*k*k*d]);
                    
                    // Blocking d: This is not as much for locality as it is for
                    // branch prediction. Blocking d reduces the number of branch 
                    // predictor misses by 4x (e.g. LeNet conv2).
                    for (int Dd_block=0; Dd_block<d; Dd_block+=D_BLOCK_SIZE)
                    {
                        // Handle edge cases
                        if (d-Dd_block < D_BLOCK_SIZE)
                        {
                            for (int Dd=Dd_block; Dd<d; ++Dd) {
                                for (int Dr=0; Dr<k; ++Dr) {
                                    for (int Dc=0; Dc<k; ++Dc) {
                                        if ( (r*s-p+Dr) >= 0 && (r*s-p+Dr) < n && (c*s-p+Dc) >= 0 && (c*s-p+Dc) < n ) {
                                            current_row[Dd*k*k + Dr*k + Dc] = D[bi*n*n*d + Dd*n*n + (r*s-p+Dr)*n + (c*s-p+Dc)];
                                        } else {
                                            current_row[Dd*k*k + Dr*k + Dc] = 0;
                                        }
                                    }
                                }
                            }
                        }
                        // Main block of D_BLOCK_SIZE: move the Dd check inside
                        // We lose the locality of writing back to current_row this way.
                        // However, k is usually small. TODO: special-case large k.
                        else
                        {
                            for (int Dr=0; Dr<k; ++Dr) {
                                for (int Dc=0; Dc<k; ++Dc) {
                                    for (int Dd=Dd_block; Dd<Dd_block+D_BLOCK_SIZE; ++Dd) {
                                        if ( (r*s-p+Dr) >= 0 && (r*s-p+Dr) < n && (c*s-p+Dc) >= 0 && (c*s-p+Dc) < n ) {
                                            current_row[Dd*k*k + Dr*k + Dc] = D[bi*n*n*d + Dd*n*n + (r*s-p+Dr)*n + (c*s-p+Dc)];
                                        } else {
                                            current_row[Dd*k*k + Dr*k + Dc] = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

// Below is the unoptimized forward lowering code
// The above code is the same except it handles special cases
// e.g. to support blocking.
// The original code below is a good starting-point for optimization
// and for understanding the optimized code above.

#else // No FW_LOWER_OPTIMIZATIONS

    // TODO: Can re-order loops below for better blocking
    
    // Top of equation 4 (for bi, r and c in ...)
    for (int bi=0; bi<b; ++bi) {
        for (int r=0; r<m; ++r) {
            for (int c=0; c<m; ++c) {
            
                // Get the row of D_lowered
                // This is the left hand side of equation 4
                float *current_row = &(D_lowered[(bi*m*m + r*m + c)*k*k*d]);

                // Calculate the right hand side of equation 4 by concatenating
                // the k*k*d cube of D
                for (int Dd=0; Dd<d; ++Dd) {
                    for (int Dr=0; Dr<k; ++Dr) {
                        // Do edge checks for padding
                        // TODO: Hoist these out
                        if ( (r*s-p+Dr) >= 0 && (r*s-p+Dr) < n ) {
                            for (int Dc=0; Dc<k; ++Dc) {
                                if ( (c*s-p+Dc) >= 0 && (c*s-p+Dc) < n ) {
                                    current_row[Dd*k*k + Dr*k + Dc] = D[bi*n*n*d + Dd*n*n + (r*s-p+Dr)*n + (c*s-p+Dc)];
                                } else {
                                    current_row[Dd*k*k + Dr*k + Dc] = 0;
                                }
                            }
                        } else {
                            for (int Dc=0; Dc<k; ++Dc) {
                                current_row[Dd*k*k + Dr*k + Dc] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
#endif // FW_LOWER_OPTIMIZATIONS
}

// Note: lower_cube and also inverse_lower_cube are special-case functions, i.e.
// they do not use parallel map + kernel callbacks. They could use that interface
// but it may be easier for fusion to keep them separate.
void CPUDriver::inverse_lower_cube(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const struct _inverse_lower_cube_arg_helper args) {

  const size_t ow = args.data_output_width;
  const size_t oh = args.data_output_height;
  const size_t k = args.kernel_size;
  const size_t s = args.stride;
  const size_t p = args.padding;
  const size_t iR = args.iR;
  const size_t iC = args.iC;
  const size_t iD = args.iD;
  const unsigned int iB = args.iB;
  float * const input_data = (float *) dst->ptr;
  const float * const output_data = (float *) src->ptr;
  // SHADJIS TODO: Do some experiments reordering these loops
  // Can add blocking too like in FW lower
  for (size_t id = 0; id < iD; ++id) {
    for (size_t ib = 0; ib < iB; ++ib) {
      for (size_t kr = 0; kr < k; ++kr) {
        for (size_t cr = 0; cr < ow; ++cr) {
          for (size_t kc = 0; kc < k; ++kc) {
            for (size_t cc = 0; cc < oh; ++cc) {
              // Unsigned so no need to check < 0. SHADJIS TODO: Try int
              // if ((cr*s + kr - p) >= 0 && (cr*s + kr - p) < iR && (cc*s + kc - p) >= 0 && (cc*s + kc - p) < iC) {
              if ((cr*s + kr - p) < iR && (cc*s + kc - p) < iC) {
                input_data[id*iR*iC + (cc*s + kc - p) + (cr*s + kr - p)*iC + ib*iR*iC*iD] += output_data[
                  id*k*k*iB*oh*ow + 
                  kr*k*iB*oh*ow + 
                  kc*iB*oh*ow + 
                  ib*oh*ow + 
                  oh*cr + 
                  cc
                ];
              }
            }
          }
        }
      }
    }
  }
}

void CPUDriver::backward_bias(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const int fmap_size, const int depth, const int batch_size,
    const float *const device_ones){
    
    // SHADJIS TODO: For the GPU, I had to rewrite backward_bias because it did
    // not work with parallel_map (see comment in ConvolutionBridge_impl.hxx).
    // Since CPU parallel_map is serial, it didn't need to be rewritten. If
    // that changes, implement the parallel version here.
    assert(false);

}

void CPUDriver::maxpool_forward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _pool_forward_arg_helper args) {

  const int iR = args.iR;
  const int iC = args.iC;
  const int oR = args.oR;
  const int oC = args.oC;
  const int D  = args.D;
  const int B  = args.B;
  const int k  = args.kernel_size;
  const int s  = args.stride;

  for (int i=0; i<D*B; ++i) {
    int * const max_index  = args.max_index + i*oR*oC;
    float * const output_data = ((float*) dst->ptr) + i*oR*oC;
    const float * const input_data  = ((float*) src->ptr) + i*iR*iC;
    for (int ph = 0; ph < oR; ++ph) {
      const int h_end = min(ph*s + k, iR);
      for (int pw = 0; pw < oC; ++pw) {
        const int w_end = min(pw*s + k, iC);
        for (int h = ph*s; h < h_end; ++h) {
          for (int w = pw*s; w < w_end; ++w) {
            max_index[ph*oC + pw] = input_data[h*iC + w] > output_data[ph*oC + pw] ?
              h*iC + w : max_index[ph*oC + pw];
            output_data[ph*oC + pw] = input_data[h*iC + w] > output_data[ph*oC + pw] ?
              input_data[h*iC + w] : output_data[ph*oC + pw];
          }
        }
      }
    }
  }
}

void CPUDriver::maxpool_backward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _pool_backward_arg_helper args) {

  const int iR = args.iR;
  const int iC = args.iC;
  const int oR = args.oR;
  const int oC = args.oC;
  const int D  = args.D;
  const int B  = args.B;
  const int * const max_index  = args.max_index;
  const float * const output_grad = (float*) dst->ptr;
  float * const input_grad  = (float*) src->ptr;
  for (int i=0; i<D*B; ++i) {
    for (int ph = 0; ph < oR; ++ph) {
      for (int pw = 0; pw < oC; ++pw) {
        input_grad[i*iR*iC + max_index[i*oR*oC + ph*oC + pw]] += output_grad[i*oR*oC + ph*oC + pw];
      }
    }
  }
}

void CPUDriver::lrn_forward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _lrn_forward_arg_helper args, const struct _lrn_forward_normalize_arg_helper args2) {

    assert(false);
}

void CPUDriver::lrn_backward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _lrn_backward_arg_helper args) {

    assert(false);
}

template<FUNC_IDX_MAPPING f_dst_pos, FUNC_MM_MAPPING func>
void CPUDriver::parallel_map(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry) {
  char * p_dst = (char*) dst->ptr;
  char * p_src = (char*) src->ptr;
  const size_t src_size = src->size_in_byte;
  for (size_t i=0; i<src_size; i+=src_skip) {
    func(&p_dst[f_dst_pos(i, f_dst_pos_curry->ptr)], &p_src[i], func_curry->ptr, f_dst_pos(i, f_dst_pos_curry->ptr));
  }
}

void CPUDriver::math_saxpy(const float alpha, DeviceMemoryPointer * X, DeviceMemoryPointer * Y) const {
  math_saxpby(X->size_in_byte/sizeof(float), alpha, (float *) X->ptr, 1.0, (float *) Y->ptr);
}

void CPUDriver::math_saxpby(const float alpha, DeviceMemoryPointer * X, const float beta, DeviceMemoryPointer * Y) const {
  math_saxpby(X->size_in_byte/sizeof(float), alpha, (float *) X->ptr, beta, (float *) Y->ptr);
}

void CPUDriver::math_saxpy(const int nItems, const float alpha, float * X, float * Y) const {
  math_saxpby(nItems, alpha, X, 1.0, Y);
}

void CPUDriver::math_saxpby(const int N, const float alpha, float * X, const float beta, float * Y) const {
#ifdef _USE_OPENBLAS
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
#elif _USE_ATLAS
  catlas_saxpby(N, alpha, X, 1, beta, Y, 1);
#elif _VANILLA_BLAS
#warning "[PERFORMANCE WARNING] Using hand-written BLAS calls. Hope you have a good compiler!"
  for (int i = N; i > 0; X++, Y++, --i) {
    *Y = alpha**X + beta* *Y;
  }
#else
#error "Select a BLAS framework."
#endif
}

void CPUDriver::set_num_threads(const int nThreads) {
#ifdef _USE_OPENBLAS
  openblas_set_num_threads(nThreads);
#elif _USE_ATLAS

#elif _VANILLA_BLAS

#else
#error "Select a BLAS framework."
#endif
}

void CPUDriver::sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB,
    int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
    float beta, float * pC, int LDC) {

  cblas_sgemm(order, TA, TB, M, N, K, alpha,
      pA, LDA,
      pB, LDB,
      beta, pC, LDC);
}

void CPUDriver::sgemm_new(const CBLAS_TRANSPOSE TA, const CBLAS_TRANSPOSE TB,
    const int M, const int N, const int K, const float alpha,
    const float * pA, const float * pB, const float beta, float * pC) {
      
  int lda = (TA == CblasNoTrans) ? K : M;
  int ldb = (TB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TA, TB, M, N, K, alpha, pA, lda, pB,
      ldb, beta, pC, N);

}

template<FUNC_STRANSFORM func>
void CPUDriver::sapply(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry) {
  const size_t n_element = dst->size_in_byte/sizeof(float);
  float * p = (float*) dst->ptr;
  for (size_t i=0;i<n_element;i++) {
    *(p) = func(*(p), func_curry->ptr);
    p++;
  }
}

template<FUNC_SREDUCE func>
void CPUDriver::selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1,
    DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry) {
  const size_t n_element = dst->size_in_byte / sizeof(float);
  float * const p_dst = (float*) dst->ptr;
  const float * const p_src1 = (float*) src1->ptr;
  const float * const p_src2 = (float*) src2->ptr;
  for (size_t i = 0; i < n_element; i++) {
    p_dst[i] = func(p_src1[i], p_src2[i], func_curry->ptr);
  }
}

void CPUDriver::sinitialize_xavier(DeviceMemoryPointer *arr, const size_t n_batch) {
  const size_t n_arr_elements = arr->size_in_byte / sizeof(float);
  const size_t fan_in = n_arr_elements / n_batch;
  const float scale = sqrt(3.0 / fan_in);

  mt19937 gen(rd());
  uniform_real_distribution<float> uni(-scale, scale);
  float * temp = (float*) arr->ptr;
  for (unsigned int i = 0; i < n_arr_elements; i++) {
    temp[i] = uni(gen);
  }
}

void CPUDriver::sbernoulli_initialize(DeviceMemoryPointer *arr, const float p) {
  const size_t n_arr_elements = arr->size_in_byte / sizeof(float);

  mt19937 gen(rd());
  bernoulli_distribution bern(p);
  float * temp = (float*) arr->ptr;
  for (unsigned int i = 0; i < n_arr_elements; i++) {
    temp[i] = bern(gen);
  }
}

void CPUDriver::sgaussian_initialize(DeviceMemoryPointer *arr, const float mean, const float std_dev) {
  const size_t n_arr_elements = arr->size_in_byte / sizeof(float);
  mt19937 gen(rd());
  normal_distribution<float> gaussian(mean, std_dev);
  float * temp = (float*) arr->ptr;
  for (unsigned int i = 0; i < n_arr_elements; i++) {
    temp[i] = gaussian(gen);
  }
}

void CPUDriver::sconstant_initialize(DeviceMemoryPointer *arr, const float value) {
  const size_t n_arr_elements = arr->size_in_byte / sizeof(float);
  float * const temp = (float*) arr->ptr;
  for (unsigned int i = 0; i < n_arr_elements; i++) {
    temp[i] = value;
  }
}

void * CPUDriver::choose_ptr(void * host, void * device) {
  return host;
}

/**
 * This is necessary for template to be instantiated.
 */
template void CPUDriver::pmap2d_read_coalesce<_fpmap_id,_fmap_lower>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, const struct PMapHelper args);

template void CPUDriver::lower_cube<_fpmap_id,_fmap_lower>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, const struct PMapHelper args);

template void CPUDriver::pmap2d_read_coalesce<_fpmap_id,_fmap_remap>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, const struct PMapHelper args);

/** All template instantiations for parallel_map **/
template void CPUDriver::parallel_map<_f_idx_strid4_copy,_f_strid4_copy>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// inverse_lower_cube
template void CPUDriver::parallel_map<_f_src_to_dst_inverse_lower_cube,_f_inverse_lower_cube>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Conv/FC Bias forward
template void CPUDriver::parallel_map<_f_src_to_dst_bias_forward,_f_bias_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Conv/FC Bias backward
template void CPUDriver::parallel_map<_f_src_to_dst_bias_backward,_f_bias_backward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// ReLU forward
template void CPUDriver::parallel_map<_f_src_to_dst_relu_forward,_f_relu_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// ReLU backward
template void CPUDriver::parallel_map<_f_src_to_dst_relu_backward,_f_relu_backward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Dropout forward train
template void CPUDriver::parallel_map<_f_src_to_dst_dropout_forward,_f_dropout_forward_train>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Dropout forward test
template void CPUDriver::parallel_map<_f_src_to_dst_dropout_forward,_f_dropout_forward_test>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Pool forward
template void CPUDriver::parallel_map<_f_src_to_dst_pool_forward,_f_pool_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Pool backward
template void CPUDriver::parallel_map<_f_src_to_dst_pool_backward,_f_pool_backward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// LRN forward
template void CPUDriver::parallel_map<_f_src_to_dst_lrn_forward,_f_lrn_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// LRN forward normalize
template void CPUDriver::parallel_map<_f_src_to_dst_lrn_forward,_f_lrn_forward_normalize>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// LRN backward
template void CPUDriver::parallel_map<_f_src_to_dst_lrn_backward,_f_lrn_backward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Softmax forward
template void CPUDriver::parallel_map<_f_src_to_dst_softmax_forward,_f_softmax_forward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);
// Softmax backward
template void CPUDriver::parallel_map<_f_src_to_dst_softmax_backward,_f_softmax_backward>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src, size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry,
    DeviceMemoryPointer * const func_curry);

template void CPUDriver::sapply<_f_add_one>(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry);

template void CPUDriver::sapply<_f_set>(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry);

template void CPUDriver::selementwise_reduce2<_f_reduce>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src1, DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);

template void CPUDriver::selementwise_reduce2<_f_reduce_mul>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src1, DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);

template void CPUDriver::selementwise_reduce2<_f_reduce_tanhgrad>(DeviceMemoryPointer * dst,
    DeviceMemoryPointer * src1, DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry);

