
#ifndef _KERNEL_LOWERING_HXX
#define _KERNEL_LOWERING_HXX

#include "lowering.h"

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_inverse_lower_cube(size_t src_pos, void * const _arg) {
  const _inverse_lower_cube_arg_helper * const arg = (_inverse_lower_cube_arg_helper *) _arg;
  return (src_pos / arg->kernel_size / arg->kernel_size / arg->data_output_width / \
      arg->data_output_height / arg->iB) * arg->iR * arg->iC;
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_inverse_lower_cube(void * input, void * output, void * const _arg, const size_t dst_index) {

  const _inverse_lower_cube_arg_helper * const arg = (_inverse_lower_cube_arg_helper *) _arg;
  const size_t data_output_width = arg->data_output_width;
  const size_t data_output_height = arg->data_output_height;
  const size_t kernel_size = arg->kernel_size;
  const size_t stride = arg->stride;
  const size_t padding = arg->padding;
  const int iR = arg->iR;
  const int iC = arg->iC;
  const int iD = arg->iD;
  const int iB = arg->iB;

  float * const input_data = (float *) input;
  const float * const output_data = (float *) output;

  size_t out_index = 0;
  // First, we iterate over K * K, which is the number of rows in the output gradient
  // cube. (Remember: the output gradient cube has dimensions K * K * iD x oR * oC * iB x 1 x 1,
  // where oR and oC do NOT refer the variables above.)
  for (size_t kr = 0; kr < kernel_size; ++kr) {
    for (size_t kc = 0; kc < kernel_size; ++kc) {

      // Then, we iterate over oR * oC * iB, the number of columns in the output gradient
      // cr and cc represent the row index and column index of the convolutional "window"
      // in the input gradient cube, which means that they must be incremented by stride
      for (size_t ib = 0; ib < iB; ++ib) {
        for (size_t cr = 0; cr < stride * data_output_width; cr += stride) {
          const int input_row_index = cr + kr - padding;

          for (size_t cc = 0; cc < stride * data_output_height; cc += stride) {
            const int input_col_index = cc + kc - padding;

            // (cr + kr - padding, cc + kc - padding) represents the index into
            // the input gradient cube. If we aren't within [0, iR) and [0, iC)
            // then we shouldn't update, because we are in the padded area
            if (input_row_index >= 0 && input_row_index < iR  &&
                input_col_index >= 0 && input_col_index < iC) {
              input_data[input_col_index + input_row_index*iC + ib*iR*iC*iD] += output_data[out_index];
            }
            // increment out_index regardless, a single cell from the output gradient cube
            // can only make a single contribution to the input gradient cube (Remember: this
            // is the *lowered* output gradient cube!)
            ++out_index;
          }
        }
      }
    }
  }
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _fpmap_id(Block2D * const output_block, const Block2D * const input_block, const PMapHelper * const args){
  output_block->r = 0;
  output_block->c = 0;
  output_block->d = 0;
  output_block->d = 0;
  output_block->dr = args->kR;
  output_block->dc = args->kC;
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline int next_multiple(int x, const int s) {
  const int m = x % s;
  x += (x > 0 && m > 0) ? s - m : -m;
  return x;
}


#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _fmap_lower(float * output, const Block2D * const output_block, const PointIn2DBlock * const input_point, const PMapHelper * const args){

  const int ir = (int) input_point->r;
  const int ic = (int) input_point->c;
  const int ib = (int) input_point->block.b;
  const int id = (int) input_point->block.d;

  const int kR = (int) args->kR;
  const int kC = (int) args->kC;
  const int iR = (int) args->sR;
  const int iC = (int) args->sC;
  const int iB = (int) args->sB;

  const int padding = (int) args->padding;
  const int stride = (int) args->stride; assert(stride > 0);


  const int output_R = (iR - kR + 2*padding + 1) / stride ;
  const int output_C = (iC - kC + 2*padding + 1) / stride ;

  const int old_o_base_col = ib * (iR-kR+1+2*padding)/stride*(iC-kC+1+2*padding)/stride;
  const int o_base_col = ib * output_R * output_C;
  
  const int o_base_row = id * kR * kC;
  const int oC = iB * output_R * output_C;
  const int old_oC = iB * (iR-kR+1+2*padding)/stride*(iC-kC+1+2*padding)/stride;
  assert(o_base_col == old_o_base_col);
  assert(oC == old_oC);

  const float input = input_point->data;

  // Starting Code computation. TODO give a real comment here.
    //int r_begin = next_multiple(std::max(ir - kR + 1 + padding, -padding), stride) - padding;
  const int r_begin = next_multiple(ir - kR + 1 + padding, stride) - padding;
  const int r_end   = std::min(ir + 1, (iR - kR + 1) + padding);
  const int i_start = (r_begin + padding) / stride;

  //int c_begin = next_multiple(std::max(ic - kC + 1 + padding, -padding), stride) - padding;
  int c_begin       = next_multiple(ic - kC + 1 + padding, stride) - padding;
  const int c_end   = std::min(ic + 1, (iC - kC + 1) + padding);
  const int j_start = (c_begin + padding) / stride;

  int rstart = (ir-kR+1) + padding + 100*stride;
  int cstart = (ic-kC+1) + padding + 100*stride;
  if(rstart % stride != 0){
    rstart += (stride-rstart%stride);
  }
  if(cstart % stride != 0){
    cstart += (stride-cstart%stride);
  }
  rstart = rstart - padding - 100*stride;
  cstart = cstart - padding - 100*stride;

  assert(rstart == r_begin);
  assert(cstart == c_begin);

  for(int r=rstart, i=i_start;r<=ir;r+=stride,++i){
    int dr = ir-r;
    assert(i == (r+padding)/stride);
    for(int c=cstart, j=j_start;c<=ic;c+=stride, j++){
      int dc = ic-c;
      assert(j == (c+padding)/stride);

      const int ocol = i*output_C+j;

      int old_ocol = (r+padding)/stride*(iC-kC+2*padding+1)/stride+(c+padding)/stride;
      assert(ocol == old_ocol);
      int orow = dr*kC+dc;

      int ocol2 = ocol + o_base_col;
      int orow2 = orow + o_base_row;
      // then write to ocol, orow

      if(c >= -padding && c < (iC-kC+1)+padding && r >= -padding && r < (iR-kR+1)+padding){
        output[ocol2 + orow2*oC] = input;

      }
    }
  }
}

#ifdef _GPU_TARGET
__host__ __device__
#endif
inline void _fmap_remap(float * output, const Block2D * const output_block, const PointIn2DBlock * const input_point, const PMapHelper * const args){
  const size_t ir = input_point->r;
  const size_t ic = input_point->c;
  const size_t ib = input_point->block.b;
  const size_t id = input_point->block.d;

  const size_t iR = args->sR;
  const size_t iC = args->sC;
  const size_t iB = args->sB;
  const size_t iD = args->sD;

  const size_t reald = (id + ib * iD) / (iB);
  const size_t realb = (id + ib * iD) % (iB);

  const float input = input_point->data;

  output[ic + ir*iC + reald*iR*iC + realb*iR*iC*iD] = input;
}

#endif
