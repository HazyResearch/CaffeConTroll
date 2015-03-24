
#ifndef _KERNEL_LOWERING_HXX
#define _KERNEL_LOWERING_HXX

#include "lowering.h"

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
  const int stride = (int) args->stride;

  const int o_base_col = ib * (iR-kR+1+2*padding)/stride*(iC-kC+1+2*padding)/stride;
  const int o_base_row = id * kR * kC;
  const int oC = iB * (iR-kR+1+2*padding)/stride*(iC-kC+1+2*padding)/stride;

  const float input = input_point->data;

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

  for(int r=rstart;r<=ir;r+=stride){
    int dr = ir-r;
    for(int c=cstart;c<=ic;c+=stride){
      int dc = ic-c;

      int ocol = (r+padding)/stride*(iC-kC+2*padding+1)/stride+(c+padding)/stride;
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










