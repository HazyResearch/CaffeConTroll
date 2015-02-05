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
MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::MaxPoolingBridge(InputLayerType * const _p_input_layer, OutputLayerType * const _p_output_layer,
    const BridgeConfig * const _bconfig)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_p_input_layer, _p_output_layer), bconfig(_bconfig) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();

  const size_t _k_size = bconfig->kernel_size;
  const size_t _stride = bconfig->stride;

#ifdef _DO_ASSERT
  assert(iD == oD); assert(iB == oB);
  assert(_stride >= _k_size); // for now, we assume that the K x K patches for
                              // max pooling never overlap
#endif

  pooled_height = static_cast<size_t>(ceil(static_cast<float>(
      iR - _k_size) / _stride)) + 1;
  pooled_width = static_cast<size_t>(ceil(static_cast<float>(
      iC  - _k_size) / _stride)) + 1;

#ifdef _DO_ASSERT
  assert(oR == pooled_height); assert(oC == pooled_width);
#endif

  // create Logical Cube to keep track of indices for max values
  max_index = new LogicalCube<size_t, Layout_CRDB>(pooled_height, pooled_width, iD, iB);
  p_output_layer->p_data_cube->reset_cube(-FLT_MAX);
  p_input_layer->p_gradient_cube->reset_cube();

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Forward direction for max pooling
 **/
template <typename DataType>
void MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::forward() {

  report_forward_last_transfer.reset();

  const LogicalCube<DataType, Layout_CRDB> * const input_data = p_input_layer->p_data_cube;
  LogicalCube<DataType, Layout_CRDB> * const output_data = p_output_layer->p_data_cube;
  for (size_t b_i = 0; b_i < iB; ++b_i) {
    for (size_t d_i = 0; d_i < iD; ++d_i) {
      const LogicalMatrix<DataType> input_data_slice = input_data->get_logical_matrix(d_i, b_i);
      LogicalMatrix<size_t> max_index_slice = max_index->get_logical_matrix(d_i, b_i);
      LogicalMatrix<DataType> output_data_slice = output_data->get_logical_matrix(d_i, b_i);

      for (size_t ph = 0; ph < pooled_height; ++ph) {
        const size_t h_start = ph * bconfig->stride;
        const size_t h_end = min(h_start + bconfig->kernel_size, iR);
        for (size_t pw = 0; pw < pooled_width; ++pw) {
          const size_t w_start = pw * bconfig->stride;
          const size_t w_end = min(w_start + bconfig->kernel_size, iC);
          const size_t pool_index = ph * pooled_width + pw;
          for (size_t h = h_start; h < h_end; ++h) {
            for (size_t w = w_start; w < w_end; ++w) {
              const size_t index = h * iC + w;
              if (input_data_slice.p_data[index] > output_data_slice.p_data[pool_index]) {
                output_data_slice.p_data[pool_index] = input_data_slice.p_data[index];
                max_index_slice.p_data[pool_index] = index;
                //cout << index << " ";
                //cout << pool_index << endl;
              }
              // if (max_index_slice.p_data[pool_index] == 0){
              //   max_index_slice.p_data[pool_index] = h_start * iC + w_start;
              // }
            }
          }
        }
      }
    }
  }

  report_forward_last_transfer.end();
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Backward direction for max pooling. (Note: we don't handle the case of max ties)
 **/
template <typename DataType>
void MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::backward() {

  report_backward_updateweight_last_transfer.reset();

  const LogicalCube<DataType, Layout_CRDB>* const input_grad = p_input_layer->p_gradient_cube;
  LogicalCube<DataType, Layout_CRDB>* const output_grad = p_output_layer->p_gradient_cube;
  //max_index->logical_print();
  for (size_t b_i = 0; b_i < iB; ++b_i) {
    for (size_t d_i = 0; d_i < iD; ++d_i) {
      const LogicalMatrix<DataType> output_grad_slice = output_grad->get_logical_matrix(d_i, b_i);
      LogicalMatrix<size_t> max_index_slice = max_index->get_logical_matrix(d_i, b_i);
      LogicalMatrix<DataType> input_grad_slice = input_grad->get_logical_matrix(d_i, b_i);

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          const size_t index = ph * pooled_width + pw;
          const size_t input_grad_index = max_index_slice.p_data[index];
          input_grad_slice.p_data[input_grad_index] += output_grad_slice.p_data[index];
        }
      }
    }
  }

  report_backward_updateweight_last_transfer.end();
  report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

template <typename DataType>
MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::~MaxPoolingBridge() {
  delete max_index;
}

#endif
