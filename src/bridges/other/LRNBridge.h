//
//  LRNBridge.h
//  moka
//
//  Created by Firas Abuzaid on 1/22/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "PhysicalOperator.h"
#include "AbstractBridge.h"

#ifndef moka_LRN_Bridge_h
#define moka_LRN_Bridge_h

//class LRNOperation : public Operation{
//public:
//	int local_size;
//	float alpha;
//	float beta;
//	bool is_across;
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
//	LRNOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map,
//				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input,
//				int _local_size=5, float _alpha=0.0001, float _beta=0.75, bool _is_across=true):
//		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
//				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
//			local_size=_local_size;
//			alpha=_alpha;
//			beta=_beta;
//			is_across=_is_across;
//	}
//
//	void backward(int batch_core, int starting_ind){
//		if(is_across){
//			for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
//				for(int ofm=0; ofm<noutput_feature_map; ofm++)
//					for(int r=0;r<nrow_output;r++)
//						for(int c=0;c<ncol_output;c++){
//							float cvalue = output[mb][ofm][r][c];
//							float cgrad = grad[mb][ofm][r][c];
//
//							int begin=max(0,ofm-local_size/2);
//							int end=min(noutput_feature_map,ofm+local_size/2);
//							for(int ifm=begin; ifm<end; ifm++)
//								grads[mb][ifm][r][c]+=2*alpha*beta*inputs[mb][ofm][r][c]/local_size*pow(pow(cvalue,beta-1),1.0/beta)*cgrad;
//						}
//		}
//		else{
//			for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
//				for(int ofm=0; ofm<noutput_feature_map; ofm++)
//					for(int r=0;r<nrow_output;r++)
//						for(int c=0;c<ncol_output;c++){
//							float cvalue = output[mb][ofm][r][c];
//							float cgrad = grad[mb][ofm][r][c];
//
//							int i_begin=max(0,r-local_size/2);
//							int i_end=min(noutput_feature_map,r+local_size/2);
//							int j_begin=max(0,c-local_size/2);
//							int j_end=min(noutput_feature_map,c+local_size/2);
//							for(int i=i_begin; i<i_end; i++)
//								for(int j=j_begin; j<j_end; j++)
//									grads[mb][ofm][i][j]+=2*alpha*beta*inputs[mb][ofm][r][c]/local_size*pow(pow(cvalue,beta-1),1.0/beta)*cgrad;
//						}
//		}
//	}
//
//	void forward(int batch_core, int starting_ind){
//		if(is_across){
//			for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
//				for(int ofm=0; ofm<noutput_feature_map; ofm++)
//					for(int r=0;r<nrow_output;r++)
//						for(int c=0;c<ncol_output;c++){
//							int begin=max(0,ofm-local_size/2);
//							int end=min(noutput_feature_map,ofm+local_size/2);
//							float sum=0;
//							for(int ifm=begin; ifm<end; ifm++)
//								sum+=inputs[mb][ifm][r][c]*inputs[mb][ifm][r][c];
//							sum=sum*alpha/local_size+1;
//							output[mb][ofm][r][c]=inputs[mb][ofm][r][c]/pow(sum,beta);
//						}
//		}
//		else{
//			for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
//				for(int ofm=0; ofm<noutput_feature_map; ofm++)
//					for(int r=0;r<nrow_output;r++)
//						for(int c=0;c<ncol_output;c++){
//							int i_begin=max(0,r-local_size/2);
//							int i_end=min(noutput_feature_map,r+local_size/2);
//							int j_begin=max(0,c-local_size/2);
//							int j_end=min(noutput_feature_map,c+local_size/2);
//							float sum=0;
//							for(int i=i_begin; i<i_end; i++)
//								for(int j=j_begin; j<j_end; j++)
//									sum+=inputs[mb][ofm][i][j]*inputs[mb][ofm][i][j];
//							sum=sum*alpha/local_size+1;
//							output[mb][ofm][r][c]=inputs[mb][ofm][r][c]/pow(sum,beta);
//						}
//		}
//	}
//};

#endif
