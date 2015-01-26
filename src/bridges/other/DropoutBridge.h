//
//  DropoutBridge.h
//  moka
//
//  Created by Firas Abuzaid on 1/22/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "PhysicalOperator.h"
#include "AbstractBridge.h"

#ifndef moka_Dropout_Bridge_h
#define moka_Dropout_Bridge_h

//class DropoutOperation : public Operation{
//public:
//	float ratio;
//	float scale;
//
//	void clear_grad(){
//		if(grads[0] != NULL)
//			for(int mb=0; mb<mini_batch_size; mb++)
//				for(int fm=0; fm<ninput_feature_map; fm++)
//					for(int r=0;r<nrow_input;r++)
//						for(int c=0; c<ncol_input;c++)
//							grads[mb][fm][r][c] = 0;
//	}
//
//	DropoutOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map,
//				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input,float _ratio):
//		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
//				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
//			ratio = _ratio;
//			scale = 1./(1.-ratio);
//
//	}
//
//	void backward(int batch_core, int starting_ind){
//		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
//			for(int ofm=0; ofm<noutput_feature_map; ofm++)
//				for(int r=0;r<nrow_output;r++)
//					for(int c=0;c<ncol_output;c++)
//						if(output[mb][ofm][r][c]!=0)
//							grads[mb][ofm][r][c]=grad[mb][ofm][r][c]*scale;
//						else
//							grads[mb][ofm][r][c]=0;
//	}
//
//	void forward(int batch_core, int starting_ind){
//		default_random_engine generator;
//		bernoulli_distribution distribution(ratio);
//		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
//			for(int fm=0; fm<noutput_feature_map; fm++)
//				for(int r=0;r<nrow_output;r++)
//					for(int c=0;c<ncol_output;c++){
//						if (distribution(generator))
//							output[mb][fm][r][c] = inputs[mb][fm][r][c]*scale;
//						else
//							output[mb][fm][r][c] = 0;
//					}
//	}
//};

#endif
