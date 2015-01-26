//
//  SoftmaxLossBridge_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/25/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_SoftmaxLossBridge_impl_hxx
#define moka_SoftmaxLossBridge_impl_hxx

template <typename DataType>
SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::SoftmaxLossBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer, const DataLabelsLogicalCubeType * const _p_data_labels)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_p_input_layer, _p_output_layer),
p_data_labels(_p_data_labels),
ldR(p_data_labels->R), ldC(p_data_labels->C),
ldD(p_data_labels->D), ldB(p_data_labels->B) {
  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(oR==iR); assert(oC==iC);
  assert(oB==iB); assert(oD==iD);
  assert(ldR==1); assert(ldC==1);
  assert(oB==ldB); assert(oD==1);
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
void SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::forward() {

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
void SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::backward() {

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

//	float*** softweights;
//	float* biases;
//	int n_label;
//	int n_input;
//
//	void clear_grad(){
//		if(grads[0] != NULL)
//			for(int mb=0; mb<mini_batch_size; mb++)
//				for(int fm=0; fm<ninput_feature_map; fm++)
//					for(int r=0;r<nrow_input;r++)
//						for(int c=0; c<ncol_input;c++)
//							grads[mb][fm][r][c] = 0;
//	}
//	SoftmaxOperation(int _mini_batch_size, int _ninput_feature_map, int _noutput_feature_map,
//				int _nrow_output, int _ncol_output, int _nrow_input, int _ncol_input):
//		Operation(_mini_batch_size, _ninput_feature_map, _noutput_feature_map,
//				_nrow_output,_ncol_output,_nrow_input,_ncol_input){
//		n_input = _ninput_feature_map;
//		n_label = _noutput_feature_map;
//
//		assert(nrow_input == 1);
//		assert(ncol_input == 1);
//		assert(nrow_output == 1);
//		assert(ncol_output == 1);
//
//		biases = new float[n_label];
//		for(int i=0;i<n_label;i++){
//			biases[i] = 0;
//		}
//		softweights = new float ** [mini_batch_size];
//		for(int mb=0; mb<mini_batch_size; mb++){
//			softweights[mb] = new float*[n_label];
//			for(int i=0;i<n_label;i++){
//				softweights[mb][i] = new float[n_input];
//				for(int j=0;j<n_input;j++){
//					softweights[mb][i][j] = (drand48()*2-1)/10;
//				}
//			}
//		}
//	}
//
//	void backward(int batch_core, int starting_ind){
//		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
//			for(int label=0;label<n_label;label++){
//				float cvalue = output[mb][label][0][0];
//				for(int i_input=0;i_input<n_input;i_input++){
//
//					float w = softweights[mb][label][i_input];
//					float x = inputs[mb][i_input][0][0];
//
//					float grad_w = (label == groundtruth[mb])*x - cvalue*x;
//					float grad_x = (label == groundtruth[mb])*w - cvalue*w;
//					// show(groundtruth[mb])
//
//					softweights[mb][label][i_input] =
//						softweights[mb][label][i_input] + STEPSIZE * grad_w;
//
//					grads[mb][i_input][0][0] += grad_x;
//
//				}
//
//				//float w = biases[label];
//				float x = 1.0;
//				float grad_w = (label == groundtruth[mb])*x - cvalue*x;
//				//float grad_x = (label == groundtruth[mb])*w - cvalue*w;
//				biases[label] = biases[label] + STEPSIZE * grad_w;
//			}
//	}
//
//	void forward(int batch_core, int starting_ind){
//		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++)
//			for(int i=0;i<n_label;i++){
//				float sum = 0.0;
//				for(int i_input=0;i_input<n_input;i_input++){
//					sum += softweights[mb][i][i_input] * inputs[mb][i_input][0][0];
//				}
//				sum += biases[i];
//				output[mb][i][0][0] = sum;
//			}
//
//		for(int mb=starting_ind; mb<starting_ind+batch_core; mb++){
//			float sum = -100000;
//			for(int i=0;i<n_label;i++){
//				sum = logadd(sum, output[mb][i][0][0]);
//			}
//			for(int i=0;i<n_label;i++){
//				output[mb][i][0][0] = exp(output[mb][i][0][0]-sum);
//			}
//		}
//	}

#endif
