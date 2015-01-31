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
SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::SoftmaxLossBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const DataLabelsLogicalCubeType * const _p_data_labels)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_p_input_layer, _p_output_layer),
p_data_labels(_p_data_labels),
ldR(p_data_labels->R), ldC(p_data_labels->C),
ldD(p_data_labels->D), ldB(p_data_labels->B) {
  this->report_forward_constructor.reset();
  this->report_forward_last_transfer.reset();
  this->report_forward_history.reset();
#ifdef _DO_ASSERT
  assert(iR==oR);  assert(iC==oC);
  assert(iB==oB);  assert(ldR==1);
  assert(ldC==1);  assert(ldD==1);
  assert(oB==ldB); //assert(oD==ldD);
#endif

  loss = DataType(0.0);

  this->report_forward_constructor.end(0, 0, 0);
}

/**
 * forward direction for Softmax Loss
 * TODO: Predictions need to be written to output layer
 **/
template <typename DataType>
void SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::forward() {

  // TODO: uncomment this when we move to OpenBLAS implementation
  // openblas_set_num_threads(run_with_n_threads);

  this->report_forward_last_transfer.reset();

  LogicalCube<DataType, Layout_CRDB> * const input_data = p_input_layer->p_data_cube;

  const DataType * const ground_truth = p_data_labels->p_data;

  for (size_t i_b = 0; i_b < iB; ++i_b) {
    const DataType * const single_input_batch = input_data->physical_get_RCDslice(i_b);
    DataType max = single_input_batch[0];
    const size_t size_of_single_batch = iR*iC*iD;
    for (size_t i = 1; i < size_of_single_batch; ++i) {
      if (single_input_batch[i] > max) {
        max = single_input_batch[i];
      }
    }
    DataType denom = DataType(0.0);
    for (size_t i = 0; i < size_of_single_batch; ++i) {
      const DataType exponentiated_val = exp(single_input_batch[i] - max);
      denom += exponentiated_val;
    }
    loss += log(denom) - single_input_batch[static_cast<int>(ground_truth[i_b])] + max;
  }

  this->report_forward_last_transfer.end();
  //this->report_forward_last_transfer.aggregate_onlystat(p_forward_gemm_kernel->this->report_last_lowering);
  //this->report_forward_last_transfer.aggregate_onlystat(p_forward_lower_connector->this->report_last_lowering);
  this->report_forward_history.aggregate(this->report_forward_last_transfer);
}

/**
 * Backward propogation for Softmax Loss
 **/
template <typename DataType>
void SoftmaxLossBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::backward() {

  // TODO: uncomment this when we move to OpenBLAS implementation
  //openblas_set_num_threads(run_with_n_threads);

  this->report_backward_updateweight_last_transfer.reset();

  // First, copy the output gradient into the input gradient
  Util::_our_memcpy(p_input_layer->p_gradient_cube->p_data,
      p_output_layer->p_gradient_cube->p_data,
      p_output_layer->p_gradient_cube->n_elements*sizeof(DataType));

  LogicalCube<DataType, Layout_CRDB> * const input_grad = p_input_layer->p_gradient_cube;
  const DataType * const ground_truth = p_data_labels->p_data;

  for (size_t i_b = 0; i_b < iB; ++i_b) {
    DataType * const single_input_batch = input_grad->physical_get_RCDslice(i_b);
    single_input_batch[static_cast<int>(ground_truth[i_b])] -= 1;
  }
  for (size_t i_b = 0; i_b < iB; ++i_b) {
    DataType * const single_input_batch = input_grad->physical_get_RCDslice(i_b);
    const size_t size_of_single_batch = iR*iC*iD;
    for (size_t i = 0; i < size_of_single_batch; ++i) {
      single_input_batch[i] *= (1.0 / iB / (iR*iC)); // borrowing Caffe's scaling (see below)
    }
  }

  // scaling from Caffe:
  //const Dtype loss_weight = top[0]->cpu_diff()[0];
  //caffe_scal(prob_.count(), loss_weight = 1 / num / spatial_dim, bottom_diff);

  this->report_backward_updateweight_last_transfer.end();
  //this->report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_element_mul_kernel->this->report_last_lowering);
  //this->report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_gemm_updategrad_kernel->this->report_last_lowering);
  //this->report_backward_updateweight_last_transfer.aggregate_onlystat(p_forward_lower_connector->this->report_last_lowering);
  //this->report_backward_updateweight_last_transfer.aggregate_onlystat(p_backward_gemm_updateweight_kernel->this->report_last_lowering);

  this->report_backward_updateweight_history.aggregate(this->report_backward_updateweight_last_transfer);
}

#endif
