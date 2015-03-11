
#include "lowering.h"

#ifndef _KERNEL_LOWERING_HXX
#define _KERNEL_LOWERING_HXX

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
	const int o_base_col = ib * (iR-kR+1)*(iC-kC+1);
	const int o_base_row = id * kR * kC;
	const int oC = iB * (iR-kR+1)*(iC-kC+1);

	const float input = input_point->data;

	for(int r=ir-kR;r<=ir;r++){
		int dr = ir-r;
		for(int c=ic-kC;c<=ic;c++){
			int dc = ic-c;
			int ocol = r*iC+c;
			int orow = dr*kC+dc;
			int ocol2 = ocol + o_base_col;
			int orow2 = orow + o_base_row;
			// then write to ocol, orow
			if(ocol >= 0 && ocol < (iR-kR+1)*(iC-kC+1) && orow >= 0 && orow < kR*kC){
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

	const size_t reald = (id * iD + ib) / (iB);
	const size_t realb = (id * iD + ib) % (iB);

	const float input = input_point->data;

	output[ ic + ir*iC + reald*iR*iC + realb*iR*iC*iD] = input;
}


#endif










