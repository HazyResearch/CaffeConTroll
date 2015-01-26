//
//  MaxPoolingBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/25/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_MaxPoolingBridge_impl_hxx
#define moka_MaxPoolingBridge_impl_hxx

template <typename DataType>
MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::MaxPoolingBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_p_input_layer, _p_output_layer)/*, stepsize(_DEFAULT_STEPSIZE) */ {
  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(oR==i1R-i2R+1); assert(oC==i1C-i2C+1);
  assert(i1D==i2D); assert(i1B==oB);
  assert(i2B==oD);
  assert(i2R==i2C);
#endif

  // TODO

  report_forward_constructor.end(0, 0, 0);
}

/**

  This function does the following:

  First Layer {iData, iModel, iGrad}
  Next Layer {oData, oModel, oGrad}

Procedure:

(1) iData -----lowering-----> LoweredData

(2) LoweredData x iModel -----------> oData

(3) oData -----non-linear func (if any)-----> oData

 **/
template <typename DataType>
void MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::forward() {

  openblas_set_num_threads(run_with_n_threads);

  report_forward_last_transfer.reset();

  //TODO

  report_forward_last_transfer.end();
  //report_forward_last_transfer.aggregate_onlystat(p_forward_gemm_kernel->report_last_lowering);
  //report_forward_last_transfer.aggregate_onlystat(p_forward_lower_connector->report_last_lowering);

  report_forward_history.aggregate(report_forward_last_transfer);
}


/**

  This function do the following.

  First Layer {iData, iModel, iGrad}
  Next Layer {oData, oModel, oGrad}

Procedure:

(1) oData element-wise-mul oGrad -------> BackPropogatedGradient

(2) Update iGrad:

(2.1) iModel x BackPropogatedGradient -----------> LoweredGradient_for_iData

(2.2) LoweredGradient_for_iData ----inverse_of_lowering----> iGrad

(3) BackPropogatedGradient x Lowered_iData * stepsize + iModel ---------> New iModel

 **/
template <typename DataType>
void MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::backward() {

  openblas_set_num_threads(run_with_n_threads);

  report_backward_updateweight_last_transfer.reset();

  // TODO

  report_backward_updateweight_last_transfer.end();
  //report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_element_mul_kernel->report_last_lowering);
  //report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_gemm_updategrad_kernel->report_last_lowering);
  //report_backward_updateweight_last_transfer.aggregate_onlystat(p_forward_lower_connector->report_last_lowering);
  //report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_gemm_updateweight_kernel->report_last_lowering);

  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

//	void clear_grad(){
//		if(grads[0] != NULL)
//			for(int mb=0; mb<mini_batch_size; mb++)
//				for(int fm=0; fm<ninput_feature_map; fm++)
//					for(int r=0;r<nrow_input;r++)
//						for(int c=0; c<ncol_input;c++)
//							grads[mb][fm][r][c] = 0;
//	}
//
//
//	MaxPoolingOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map,
//				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input, int _stride=1, int _pad=0, int _group=1):
//		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
//				_nrow_output,_ncol_output,_nrow_input,_ncol_input,_stride,_pad,_group){
//
//		// assert(nrow_input % nrow_output == 0);
//		// assert(ncol_input % ncol_output == 0);
//			// TODO: NEED WORK
//
//	}
//
//	void backward(int batch_core, int starting_ind){
//			// TODO: FLOAT (== check)
//		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
//			for(int ofm=0; ofm<noutput_feature_map; ofm++)
//				for(int r=0;r<nrow_output;r++){
//					for(int c=0;c<ncol_output;c++){
//						float cvalue = output[mb][ofm][r][c];
//						float cgrad = grad[mb][ofm][r][c];
//						bool flag=0;
//						for(int ir=r*stride;ir<r*stride+nrow_conv;ir++){
//							for(int ic=c*stride;ic<c*stride+ncol_conv;ic++){
//								if(ir>=pad && ir<nrow_input+pad && ic>=pad && ic<ncol_input+pad){
//									if(inputs[mb][ofm][ir-pad][ic-pad] == cvalue && flag==0){
//										grads[mb][ofm][ir-pad][ic-pad] += cgrad;
//										flag=1;
//									}else{
//										grads[mb][ofm][ir-pad][ic-pad] = 0;
//									}
//								}
//							}
//						}
//					}
//				}
//
//	}
//
//	void forward(int batch_core, int starting_ind){
//		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
//			for(int ofm=0; ofm<noutput_feature_map; ofm++){
//				for(int r=0;r<nrow_output;r++){
//					for(int c=0;c<ncol_output;c++){
//						float max = -10000;
//						for(int ir=r*stride;ir<r*stride+nrow_conv;ir++){
//							for(int ic=c*stride;ic<c*stride+ncol_conv;ic++){
//								if(ir>=pad && ir<nrow_input+pad && ic>=pad && ic<ncol_input+pad){
//									if(inputs[mb][ofm][ir][ic] > max){
//										max = inputs[mb][ofm][ir-pad][ic-pad];
//									}
//								}
//								else if(0 > max){
//										max = 0;
//									}
//							}
//						}
//						output[mb][ofm][r][c] = max;
//					}
//				}
//			}
//	}

#endif
