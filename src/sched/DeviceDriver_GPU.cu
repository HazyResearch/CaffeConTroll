
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <boost/random.hpp>

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
  const int p  = args.padding;
  const int s  = args.stride;
  const int oR = (iR + 2*p - kR) / s + 1;
  const int oC = (iC + 2*p - kC) / s + 1; 

  const float *data_im = src;
  const int height = iR;
  const int width = iC;
  const int kernel_h = kR;
  const int kernel_w = kC;
  const int pad_h = p;
  const int pad_w = p;
  const int stride_h = s;
  const int stride_w = s;
  const int height_col = oR;
  const int width_col = oC;
  float *data_col = dst;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < iD*oR*oC)
  {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    float *data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const float *data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

__global__ void _parallel_inverse_lower_cube(float * dst, float * src, const struct _inverse_lower_cube_arg_helper args, const int start_i){

  // Read arguments
  const int oC = args.data_output_width;
  const int oR = args.data_output_height;
  const int k = args.kernel_size;
  const int s = args.stride;
  const int p = args.padding;
  const int iR = args.iR;
  const int iC = args.iC;
  const int iD = args.iD;

  // Get the right loop element
  const int i = blockDim.x * blockIdx.x + threadIdx.x + start_i;

  if (i < iD*iR*iC)
  {
    // SHADJIS TODO: These / and % not needed if using multi-dimensional blocks
    const int c =  i / (iC * iR);
    const int h = (i / iC) % iR + p;
    const int w =  i % iC + p;
    
    const int w_col_start = (w < k) ? 0 : (w - k) / s + 1;
    const int w_col_end = device_min(w / s + 1, oC); // SHADJIS TODO: cuda has a min function
    const int h_col_start = (h < k) ? 0 : (h - k) / s + 1;
    const int h_col_end = device_min(h / s + 1, oR); // SHADJIS TODO: cuda has a min function
    
    // SHADJIS TODO: Not sure why but the way we store batches is
    // different from Caffe so this part had to be changed. Probably
    // something to do with the old/unnecessary CcT decision to flip
    // the gemm order everywhere. So instead of storing each batch
    // of src one after the other we interleave them. I think this is
    // pretty stupid but I don't feel like rewriting everything now.
    // Probably leave that for when we do fusion.
    const int offset = (c*k*k + h*k + w)*oR*oC;
    const int coeff_h_col = (1 - s*k*oR)*oC;
    const int coeff_w_col = (1 - s*oR*oC);
    float sum = 0;
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        sum += src[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    dst[i] = sum;
  }
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
    set_device();
    // pmap2d_read_coalesce<f_id, f_data>(dst, src, args);
    lower_cube_helper(dst, src, args);
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
    assert(iB==1);
    const int p  = args.padding;
    const int s  = args.stride;
    const int iR = args.sR;
    const int iC = args.sC;
    const int oR = (iR + 2*p - kr) / s + 1;
    const int oC = (iC + 2*p - kc) / s + 1; 
    const int num_parallel_threads = iD*oR*oC;
    const int numBlocks = (num_parallel_threads + threadsPerBlock - 1) / threadsPerBlock;
    cudaGetLastError(); // Reset the error status to success
    _parallel_lower_cube<<<numBlocks, threadsPerBlock>>>((float*) dst->ptr, (float *) src->ptr, args);
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "Fail to launch _parallel_lower_cube"  << "  ERROR " << err << std::endl;
      assert(false);
    }
}

__global__ void _maxpool_forward_helper(float* output, const float* input,
    const struct _pool_forward_arg_helper args){
    
    const int iR = args.iR;
    const int iC = args.iC;
    const int oR = args.oR;
    const int oC = args.oC;
    const int D  = args.D;
    const int B  = args.B;
    const int k  = args.kernel_size;
    const int s  = args.stride;
    int * const max_index  = args.max_index;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (index < D*B*oR*oC) {
      int pw = index % oC;
      const int tmp1 = index / oC;
      const int tmp2 = tmp1  / oR;
      int ph = tmp1 % oR;
      int c =  tmp2 % D;
      int n =  tmp2 / D;
      int hstart = ph * s;// - p;
      int wstart = pw * s;// - p;
      int hend = min(hstart + k, iR);
      int wend = min(wstart + k, iC);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
      float maxval = -FLT_MAX;
      int maxidx = -1;
      input += (n * D + c) * iR * iC;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          if (input[h * iC + w] > maxval) {
            maxidx = h * iC + w;
            maxval = input[maxidx];
          }
        }
      }
      output[index] = maxval;
      max_index[index] = maxidx;
    }
}

void GPUDriver::maxpool_forward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _pool_forward_arg_helper args){
    
    set_device();
    const int oR = args.oR;
    const int oC = args.oC;
    const int D  = args.D;
    const int B  = args.B;
    const int num_parallel_threads = D*B*oR*oC;
    const int numBlocks = (num_parallel_threads + threadsPerBlock - 1) / threadsPerBlock;
    cudaGetLastError(); // Reset the error status to success
    // SHADJIS TODO: Add loop to call multiple times if necessary
    _maxpool_forward_helper<<<numBlocks, threadsPerBlock>>>((float*) dst->ptr, (float *) src->ptr, args);
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "Fail to launch _maxpool_forward_helper"  << "  ERROR " << err << std::endl;
      assert(false);
    }
}

__global__ void _maxpool_backward_helper(float* output, const float* input,
    const struct _pool_backward_arg_helper args){
    
    const int iR = args.iR;
    const int iC = args.iC;
    const int oR = args.oR;
    const int oC = args.oC;
    const int D  = args.D;
    const int B  = args.B;
    const int k  = args.kernel_size;
    const int s  = args.stride;
    const int *max_index  = args.max_index;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (index < D*B*iR*iC) {
      // find out the local index
      // find out the local offset
      const int tmp1 = index / iC;
      const int tmp2 = tmp1  / iR;
      int w = index % iC;
      int h = tmp1 % iR;
      int c = tmp2 % D;
      int n = tmp2 / D;
      int phstart = (h < k) ? 0 : (h - k) / s + 1;
      int phend = min(h / s + 1, oR);
      int pwstart = (w < k) ? 0 : (w - k) / s + 1;
      int pwend = min(w / s + 1, oC);
      float gradient = 0;
      int offset = (n * D + c) * oR * oC;
      input += offset;
      max_index += offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (max_index[ph * oC + pw] == h * iC + w) {
            gradient += input[ph * oC + pw];
          }
        }
      }
      output[index] = gradient;
    }
}

void GPUDriver::maxpool_backward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _pool_backward_arg_helper args){
    
    set_device();
    const int iR = args.iR;
    const int iC = args.iC;
    const int D  = args.D;
    const int B  = args.B;
    const int num_parallel_threads = D*B*iR*iC;
    const int numBlocks = (num_parallel_threads + threadsPerBlock - 1) / threadsPerBlock;
    cudaGetLastError(); // Reset the error status to success
    // SHADJIS TODO: Add loop to call multiple times if necessary
    _maxpool_backward_helper<<<numBlocks, threadsPerBlock>>>((float *) src->ptr, (float*) dst->ptr, args);
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "Fail to launch _maxpool_backward_helper"  << "  ERROR " << err << std::endl;
      assert(false);
    }
}

__global__ void _lrn_forward_helper_fill_scale(const float* in,
    const struct _lrn_forward_arg_helper args, const struct _lrn_forward_normalize_arg_helper args2){

    const int iR = args.iR;
    const int iC = args.iC;
    const int D  = args.iD;
    const int B  = args.iB;
    const int size  = args.local_size;
    const float alpha_over_size  = args2.alpha_over_size;
    float *scale  = (float*) args2.denoms;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < B*iR*iC) {
      // find out the local offset
      int w = index % iC;
      int h = (index / iC) % iR;
      int n = index / iC / iR;
      int offset = (n * D * iR + h) * iC + w;
      int step = iR * iC;
      in += offset;
      scale += offset;
      int head = 0;
      int pre_pad = (size - 1) / 2;
      int post_pad = size - pre_pad - 1;
      float accum_scale = 0;
      // fill the scale at [n, :, h, w]
      // accumulate values
      while (head < post_pad) {
        accum_scale += in[head * step] * in[head * step];
        ++head;
      }
      // until we reach size, nothing needs to be subtracted
      while (head < size) {
        accum_scale += in[head * step] * in[head * step];
        scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
        ++head;
      }
      // both add and subtract
      while (head < D) {
        accum_scale += in[head * step] * in[head * step];
        accum_scale -= in[(head - size) * step] * in[(head - size) * step];
        scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
        ++head;
      }
      // subtract only
      while (head < D + post_pad) {
        accum_scale -= in[(head - size) * step] * in[(head - size) * step];
        scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
        ++head;
      }
    }
}

__global__ void _lrn_forward_helper_compute_output(float* out, const float* in,
    const struct _lrn_forward_arg_helper args, const struct _lrn_forward_normalize_arg_helper args2){
    
    const int iR = args.iR;
    const int iC = args.iC;
    const int D  = args.iD;
    const int B  = args.iB;
    const float beta  = args2.beta;
    float *scale  = (float*) args2.denoms;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (index < D*B*iR*iC) {
        out[index] = in[index] * pow(scale[index], float(-1*beta));
    }
}

void GPUDriver::lrn_forward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _lrn_forward_arg_helper args, const struct _lrn_forward_normalize_arg_helper args2){
    
    set_device();
    const int iR = args.iR;
    const int iC = args.iC;
    const int D  = args.iD;
    const int B  = args.iB;
    const int num_parallel_threads = B*iR*iC;
    const int numBlocks = (num_parallel_threads + threadsPerBlock - 1) / threadsPerBlock;
    cudaGetLastError(); // Reset the error status to success
    _lrn_forward_helper_fill_scale<<<numBlocks, threadsPerBlock>>>((float *) dst->ptr, args, args2);
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "Fail to launch _maxpool_forward_helper"  << "  ERROR " << err << std::endl;
      assert(false);
    }
    const int num_parallel_threads2 = D*B*iR*iC;
    const int numBlocks2 = (num_parallel_threads2 + threadsPerBlock - 1) / threadsPerBlock;
    _lrn_forward_helper_compute_output<<<numBlocks2, threadsPerBlock>>>((float*) src->ptr, (float *) dst->ptr, args, args2);
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "Fail to launch _maxpool_forward_helper"  << "  ERROR " << err << std::endl;
      assert(false);
    }
}

__global__ void _lrn_backward_helper(float* bottom_diff, const float* top_diff,
    const struct _lrn_backward_arg_helper args){

    const int height = args.oR;
    const int width = args.oC;
    const int channels  = args.oD;
    const int B  = args.oB;
    const int size  = args.local_size;
    const float alpha_over_size = args.alpha_over_size;
    const float negative_beta = -1. * args.beta;
    const float cache_ratio = (2. * alpha_over_size * args.beta);
    float *scale = (float*) args.denoms;
    float *bottom_data = (float*) args.input_data;
    float *top_data = (float*) args.output_data;

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
  
    if (index < B*width*height) {
        // find out the local offset
        int w = index % width;
        int h = (index / width) % height;
        int n = index / width / height;
        int offset = (n * channels * height + h) * width + w;
        int step = height * width;
        bottom_data += offset;
        top_data += offset;
        scale += offset;
        top_diff += offset;
        bottom_diff += offset;
        int head = 0;
        int pre_pad = size - (size + 1) / 2;
        int post_pad = size - pre_pad - 1;
        float accum_ratio = 0;
        // accumulate values
        while (head < post_pad) {
          accum_ratio += top_diff[head * step] * top_data[head * step] /
              scale[head * step];
          ++head;
        }
        // until we reach size, nothing needs to be subtracted
        while (head < size) {
          accum_ratio += top_diff[head * step] * top_data[head * step] /
              scale[head * step];
          bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
              * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
              bottom_data[(head - post_pad) * step] * accum_ratio;
          ++head;
        }
        // both add and subtract
        while (head < channels) {
          accum_ratio += top_diff[head * step] * top_data[head * step] /
              scale[head * step];
          accum_ratio -= top_diff[(head - size) * step] *
              top_data[(head - size) * step] / scale[(head - size) * step];
          bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
              * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
              bottom_data[(head - post_pad) * step] * accum_ratio;
          ++head;
        }
        // subtract only
        while (head < channels + post_pad) {
          accum_ratio -= top_diff[(head - size) * step] *
              top_data[(head - size) * step] / scale[(head - size) * step];
          bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
              * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
              bottom_data[(head - post_pad) * step] * accum_ratio;
          ++head;
        }
    }
}

void GPUDriver::lrn_backward(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _lrn_backward_arg_helper args){

    set_device();
    const int oR = args.oR;
    const int oC = args.oC;
    const int B  = args.oB;
    const int num_parallel_threads = B*oR*oC;
    const int numBlocks = (num_parallel_threads + threadsPerBlock - 1) / threadsPerBlock;
    cudaGetLastError(); // Reset the error status to success
    _lrn_backward_helper<<<numBlocks, threadsPerBlock>>>((float*) dst->ptr, (float *) src->ptr, args);
    err = cudaGetLastError();
    if(err != cudaSuccess){
      std::cout << "Fail to launch _lrn_backward_helper"  << "  ERROR " << err << std::endl;
      assert(false);
    }
}


// Note: lower_cube and also inverse_lower_cube are special-case functions, i.e.
// they do not use parallel map + kernel callbacks. They could use that interface
// but it may be easier for fusion to keep them separate.
void GPUDriver::inverse_lower_cube(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct _inverse_lower_cube_arg_helper args){

    set_device();
    const int iD = args.iD;
    const int iR = args.iR;
    const int iC = args.iC;
    const unsigned int iB = args.iB;
    assert(iB == 1);
    const int num_parallel_elements = iR*iC*iD;
    int blocksPerGrid = (num_parallel_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaGetLastError(); // Reset the error status to success
    // SHADJIS TODO: Call something like _spmap_readc instead
    // SHADJIS TODO: Added a check for too many blocks, but can also make multi-dimensional like _spmap_readc
    const int num_calls = (blocksPerGrid + max_cuda_blocks - 1) / max_cuda_blocks;
    if (num_calls > 1) {
        blocksPerGrid = max_cuda_blocks;
    }
    for (int call_counter=0; call_counter < num_calls; ++call_counter)
    {
        _parallel_inverse_lower_cube<<<blocksPerGrid, threadsPerBlock>>>((float*) dst->ptr, (float *) src->ptr, args, call_counter*max_cuda_blocks);
        err = cudaGetLastError();
        if(err != cudaSuccess){
            std::cout << "Fail to launch _parallel_inverse_lower_cube"  << "  ERROR " << err << std::endl;
            assert(false);
        }
    }
}

void GPUDriver::backward_bias(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const int fmap_size, const int depth, const int batch_size,
    const float *const device_ones){

    set_device();
    
    // Create the one constants
    const float one = 1;
    
    // We also get device_ones as an argument, a vector of ones of size fmap_size

    // cublas expects col major, so we change the parameters accordingly
    cublasOperation_t ta = CUBLAS_OP_T;
    
    // Call for each batch
    // SHADJIS TODO: Note this is different from the inverse bias call in fully connected.
    // There we can do all batches in a single GEMV because the output fmap size is 1x1,
    // i.e. we only have D and B dimensions so we can put all the batches together in memory
    // by transposing. But when output fmap size oR*oC != 1x1, we use multiple GEMVs.
    for (int ib=0; ib < batch_size; ++ib)
    {
        // SHADJIS TODO: Call DeviceDriver_GPU::sgemv instead
        status = cublasSgemv(handle, ta, fmap_size, depth, &one, (float *) (src->ptr) + ib*fmap_size*depth,
            fmap_size, device_ones, 1, &one, (float *) dst->ptr, 1);
    }
    err = cudaGetLastError();
    assert(err == cudaSuccess);
    assert(status == CUBLAS_STATUS_SUCCESS);
}

// SHADJIS TODO: This is just a gemv call, could call p_driver->sgemv() directly from
// fc bridge rather than call this
void GPUDriver::backward_bias_fc(DeviceMemoryPointer * bias, DeviceMemoryPointer * output,
    const int D, const int B, const float *const device_ones){

    sgemv(CblasTrans, B, D, (float) 1., (float *) (output->ptr),
        device_ones, (float) 0., (float *) (bias->ptr));
}

__global__ void _fw_bias_helper(float * bias, float * output, const int fmap_size, const int depth, const int batch_size){

  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int d = (i % (fmap_size*depth)) / fmap_size;
  if ( i < fmap_size*depth*batch_size ) {
    output[i] += bias[d];
  }
}

void GPUDriver::forward_bias(DeviceMemoryPointer * dst, DeviceMemoryPointer * src,
    const int fmap_size, const int depth, const int batch_size){

    set_device();
    // One way to do this is to make a ones vector of size fmap_size, do a GEMM 
    // with the bias vector to get depth x fmap_size, and then add this within
    // the GEMM by passing beta=1
    // But that handles each batch serially
    // Another way is to just add the bias term to the correct index
    // SHADJIS TODO: I'll try that first and switch to BLAS if slow
    
	cudaGetLastError(); // Reset the error status to success
	int n_elements =  fmap_size * depth * batch_size;
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;

	// Check if we are trying to initialize something huge. In that case call multiple times.
	// SHADJIS TODO: Could make multi-dimensional like _spmap_readc instead
	const int num_calls = (blocksPerGrid + max_cuda_blocks - 1) / max_cuda_blocks;
	if (num_calls > 1) {
		blocksPerGrid = max_cuda_blocks;
	}
	for (int call_counter=0; call_counter < num_calls; ++call_counter)
	{
		_fw_bias_helper<<<blocksPerGrid, threadsPerBlock>>>((float*) dst->ptr /*bias*/, (float*) src->ptr /*output*/,
            fmap_size, depth, batch_size);
		err = cudaGetLastError();
		if(err != cudaSuccess){
			std::cout << "Fail to launch _fw_bias_helper" << "  ERROR " << err << std::endl;
			assert(false);
		}
	}
	err = cudaGetLastError();
	assert(err == cudaSuccess);

}

// SHADJIS TODO: Why is the interface for this is different from parallel_map?
// Here we pass in the args directly whereas parallel_map gets pointers to
// the args already allocated on the device
template<FPMAP_ID f_id, FPMAP_DATA_READC f_data>
void GPUDriver::pmap2d_read_coalesce(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
    const struct PMapHelper args){

    set_device();
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
}


GPUDriver::GPUDriver(){
    set_device();
    
    // Initialize curand
    curand_err = curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (curand_err != CURAND_STATUS_SUCCESS) {
      std::cout << "Failed curandCreateGenerator, error = " << curand_err << std::endl;
      assert(false);
    }
    curand_err = curandSetPseudoRandomGeneratorSeed(curand_gen, time(NULL)); // rd is std::random_device defined in util()
    if (curand_err != CURAND_STATUS_SUCCESS) {
      std::cout << "Failed curandSetPseudoRandomGeneratorSeed, error = " << curand_err << std::endl;
      assert(false);
    }
}

GPUDriver::~GPUDriver(){
    set_device();
    
    curandDestroyGenerator(curand_gen);
    if (curand_err != CURAND_STATUS_SUCCESS) {
      std::cout << "Failed curandDestroyGenerator, error = " << curand_err << std::endl;
      assert(false);
    }
}

DeviceMemoryPointer * GPUDriver::get_device_pointer(void * ptr, size_t size_in_byte){
	// TODO: This has memory leak! Refactor it!
	return new DeviceMemoryPointer_Local_GPURAM(gpu_id, ptr, size_in_byte);
}

void GPUDriver::malloc(DeviceMemoryPointer * dst){
    set_device();
    err = cudaMalloc((void**)&dst->ptr, dst->size_in_byte);
    if(err != cudaSuccess){
      std::cout << "\nFailed cudaMalloc"  << "  ERROR " << err << "   " << cudaGetErrorString(err) << " on GPU " << gpu_id << std::endl;
      if (err == cudaErrorMemoryAllocation) {
        std::cout << "This is usually because your network is too large for the number of GPUs.\n";
        std::cout << "You can add partitions on the CPU or on other GPUs.\n\n";
      } 
      assert(false);
    }
}

void GPUDriver::free(DeviceMemoryPointer * dst){
    set_device();
	err = cudaFree(dst->ptr);
	if(err != cudaSuccess){
	  std::cout << "Failed cudaFree"  << "  ERROR " << err << std::endl;
	  assert(false);
	}
}

void GPUDriver::memcpy(DeviceMemoryPointer * dst, DeviceMemoryPointer * src){
    set_device();
	#ifdef _DO_ASSERT
	assert(dst->size_in_byte == src->size_in_byte);
	#endif
	if(src->type == DEVICEMEMORY_LOCAL_RAM){
  		err = cudaMemcpy(dst->ptr, src->ptr, dst->size_in_byte, cudaMemcpyHostToDevice);
	}else if(dst->type == DEVICEMEMORY_LOCAL_RAM){
  		err = cudaMemcpy(dst->ptr, src->ptr, dst->size_in_byte, cudaMemcpyDeviceToHost);
	}else{
		err = cudaMemcpy(dst->ptr, src->ptr, dst->size_in_byte, cudaMemcpyDeviceToDevice);
	}
	if(err != cudaSuccess){
	  std::cout << "Failed cudaMemcpy"  << "  ERROR " << err << std::endl;
	  assert(false);
	}
}

void GPUDriver::memset(DeviceMemoryPointer * dst, const char value){
    set_device();
	#ifdef _DO_ASSERT
	assert(dst->type==DEVICEMEMORY_LOCAL_RAM);
	#endif
	err = cudaMemset(dst->ptr, value, dst->size_in_byte);
	if(err != cudaSuccess){
	  std::cout << "Failed cudaMemset"  << "  ERROR " << err << std::endl;
	  assert(false);
	}
}

template<FUNC_IDX_MAPPING f_dst_pos, FUNC_MM_MAPPING func>
void GPUDriver::parallel_map(DeviceMemoryPointer * dst, DeviceMemoryPointer * src, 
size_t src_skip, DeviceMemoryPointer * const f_dst_pos_curry, DeviceMemoryPointer * const func_curry){

    set_device();
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
	cudaFree(d_func_curry);
	cudaFree(d_idx_func_curry);

}

void GPUDriver::math_saxpy(const float alpha, DeviceMemoryPointer * X, DeviceMemoryPointer * Y) const { 
    set_device();
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
  set_device();
  cublasStatus_t status = cublasSaxpy(handle, nElements, &alpha, X, 1, Y, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);
}

template<FUNC_STRANSFORM func>
void GPUDriver::sapply(DeviceMemoryPointer * dst, DeviceMemoryPointer * const func_curry){
    set_device();
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
		_sapply<func><<<blocksPerGrid, threadsPerBlock>>>((float*) (dst->ptr) + call_counter*blocksPerGrid*threadsPerBlock, n_elements, d_func_curry);
		err = cudaGetLastError();
		if(err != cudaSuccess){
			std::cout << "Fail to launch _sapply" << "  ERROR " << err << std::endl;
			assert(false);
		}
		n_elements -= blocksPerGrid*threadsPerBlock; // Decrement #elements left to process
	}
	err = cudaGetLastError();
	assert(err == cudaSuccess);

	cudaFree(d_func_curry);
}

void GPUDriver::math_saxpby(const float alpha, DeviceMemoryPointer * X, const float beta, DeviceMemoryPointer * Y) const { 
  set_device();
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
  set_device();
  cublasStatus_t status = cublasSscal(handle, nElements, &beta, Y, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);

  status = cublasSaxpy(handle, nElements, &alpha, X, 1, Y, 1);
  assert(status == CUBLAS_STATUS_SUCCESS);

}

void GPUDriver::set_num_threads(const int nThreads) { 
  // SHADJIS TODO: Can implement this on GPU but not really needed, mostly just for CPU
}

__global__ void _L1_update_helper (const int n_elements, float * const p_gradient,
    const float lambda, const float * const p_model) {
    
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_elements) {
      p_gradient[i] += lambda * (p_model[i] > 0 ? 1 : -1);
    }
}

void GPUDriver::L1_update(const int n_elements, float * const p_gradient,
    const float lambda, const float * const p_model) {

    set_device();
	int blocksPerGrid = (n_elements + threadsPerBlock - 1) / threadsPerBlock;
	const int num_calls = (blocksPerGrid + max_cuda_blocks - 1) / max_cuda_blocks;
	if (num_calls > 1) {
		blocksPerGrid = max_cuda_blocks;
	}
    int n_elements_left = n_elements;
	for (int call_counter=0; call_counter < num_calls; ++call_counter)
	{
        const int offset = call_counter*blocksPerGrid*threadsPerBlock;
		_L1_update_helper<<<blocksPerGrid, threadsPerBlock>>>(n_elements_left, p_gradient + offset, lambda, p_model + offset);
		err = cudaGetLastError();
		if(err != cudaSuccess){
			std::cout << "Fail to launch _L1_update_helper" << "  ERROR " << err << std::endl;
			assert(false);
		}
		n_elements_left -= blocksPerGrid*threadsPerBlock; // Decrement #elements left to process
	}
	err = cudaGetLastError();
	assert(err == cudaSuccess);
}

void GPUDriver::sgemm(const enum CBLAS_ORDER order, CBLAS_TRANSPOSE TA, CBLAS_TRANSPOSE TB, 
    int M, int N, int K, float alpha, float * pA, int LDA, float * pB, int LDB,
    float beta, float * pC, int LDC){
  
    set_device();
    
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

		//cudaDeviceSynchronize();
		err = cudaGetLastError();
		assert(err == cudaSuccess);

		assert(status == CUBLAS_STATUS_SUCCESS);

	}else if(TA == CblasTrans && TB == CblasNoTrans){

		cublasOperation_t ta = CUBLAS_OP_T;
		cublasOperation_t tb = CUBLAS_OP_N;

		// cublas expects col major, so we change the parameters accordingly
		status = cublasSgemm(handle, tb, ta, N, M, K, &alpha, 
			pB, N, pA, M, &beta, pC, N); 

		//cudaDeviceSynchronize();
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

		//cudaDeviceSynchronize();
		err = cudaGetLastError();
		assert(err == cudaSuccess);

		assert(status == CUBLAS_STATUS_SUCCESS);

	}else if(TA == CblasTrans && TB == CblasTrans){

		cublasOperation_t ta = CUBLAS_OP_T;
		cublasOperation_t tb = CUBLAS_OP_T;

		// cublas expects col major, so we change the parameters accordingly
		status = cublasSgemm(handle, tb, ta, N, M, K, &alpha, 
			pB, K, pA, M, &beta, pC, N); 

		//cudaDeviceSynchronize();
		err = cudaGetLastError();
		assert(err == cudaSuccess);

		assert(status == CUBLAS_STATUS_SUCCESS);

	}else{
		assert(false);
	}

}

// Easier / consistent interface than other 
void GPUDriver::sgemm_new(const CBLAS_TRANSPOSE TA, const CBLAS_TRANSPOSE TB, 
    const int M, const int N, const int K, const float alpha,
    const float * pA, const float * pB, const float beta, float * pC) {
  
    set_device();

    int lda = (TA == CblasNoTrans) ? K : M;
    int ldb = (TB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA =
        (TA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
        (TB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    status = cublasSgemm(handle, cuTransB, cuTransA,
        N, M, K, &alpha, pB, ldb, pA, lda, &beta, pC, N);

    err = cudaGetLastError();
    assert(err == cudaSuccess);
    assert(status == CUBLAS_STATUS_SUCCESS);
}

void GPUDriver::sgemv(const CBLAS_TRANSPOSE TA, const int M, const int N,
    const float alpha, const float * pA, const float * px, const float beta,
    float * py) {
  
    set_device();

    cublasOperation_t cuTransA =
        (TA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    status = cublasSgemv(handle, cuTransA, N, M, &alpha,
        pA, N, px, 1, &beta, py, 1);

    err = cudaGetLastError();
    assert(err == cudaSuccess);
    assert(status == CUBLAS_STATUS_SUCCESS);
}

template<FUNC_SREDUCE func>
void GPUDriver::selementwise_reduce2(DeviceMemoryPointer * dst, DeviceMemoryPointer * src1, 
DeviceMemoryPointer * src2, DeviceMemoryPointer * const func_curry){ 
    set_device();

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
	//cudaDeviceSynchronize();
	err = cudaGetLastError();
	assert(err == cudaSuccess);

    cudaFree(d_func_curry);
}

/**
* This function is called only once. So its speed does not matter.
* TODO: Wrap this up with CURAND.
**/
void GPUDriver::sinitialize_xavier(DeviceMemoryPointer *arr, const size_t n_batch) {
    set_device();
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

    set_device();
    const size_t n_arr_elements = arr->size_in_byte / sizeof(float);
    
    std::random_device rd;
	std::mt19937 gen(rd());
    
	float * temp = new float[n_arr_elements];
    boost::bernoulli_distribution<float> random_distribution(p);
    boost::variate_generator<mt19937, boost::bernoulli_distribution<float> >
      variate_generator(gen, random_distribution);
    for (size_t i = 0; i < n_arr_elements; ++i) {
      temp[i] = variate_generator();
    }
    
	cudaMemcpy(arr->ptr, temp, arr->size_in_byte, cudaMemcpyHostToDevice);
	delete[] temp;
}


void GPUDriver::rand_uint_initialize(unsigned int * buf, const int n) {

    set_device();
    
    // Generate random unsigned ints
    curand_err = curandGenerate(curand_gen, buf, n);
    if (curand_err != CURAND_STATUS_SUCCESS) {
      std::cout << "Failed curandGenerate, error = " << curand_err << std::endl;
      assert(false);
    }
}

/**
* This function is called only once. So its speed does not matter.
* TODO: Wrap this up with CURAND.
**/
void GPUDriver::sgaussian_initialize(DeviceMemoryPointer *arr, const float mean, const float std_dev) {
    set_device();
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
    set_device();
    DeviceMemoryPointer_Local_RAM pvalue((void*)&value, sizeof(float));
    sapply<__sconstant_initialize_helper>(arr, &pvalue);
}

void * GPUDriver::choose_ptr(void * host, void * device){
	return device;
}

void GPUDriver::device_sync() {
  set_device();
  cudaDeviceSynchronize();
}

void GPUDriver::init_thread() {
  set_device();
  // SHADJIS TODO: Might not need to create this for every thread
  cublasCreate(&handle);
}

void GPUDriver::destroy_thread() {
  set_device();
  cublasDestroy(handle);
}

void GPUDriver::set_device() const {
  cudaError_t d_err = cudaGetLastError(); // Reset error
  cudaSetDevice(gpu_id);
  d_err = cudaGetLastError();
  if(d_err != cudaSuccess){
    std::cout << "Fail to set device " << gpu_id << "  ERROR " << d_err << std::endl;
    assert(false);
  }
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

