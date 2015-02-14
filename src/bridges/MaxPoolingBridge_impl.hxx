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
MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::MaxPoolingBridge(InputLayerType * const _p_input_layer,
    OutputLayerType * const _p_output_layer, const cnn::LayerParameter * const _layer_param)
: AbstractBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>(_p_input_layer, _p_output_layer, _layer_param) {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();

  kernel_size = layer_param->pooling_param().kernel_size();
  stride = layer_param->pooling_param().stride();

#ifdef _DO_ASSERT
  assert(iD == oD); assert(iB == oB);
#endif

  pooled_height = static_cast<size_t>(ceil(static_cast<float>(
      iR - kernel_size) / stride)) + 1;
  pooled_width = static_cast<size_t>(ceil(static_cast<float>(
      iC  - kernel_size) / stride)) + 1;

#ifdef _DO_ASSERT
  assert(oR == pooled_height); assert(oC == pooled_width);
#endif

  // create Logical Cube to keep track of indices for max values
  max_index = new LogicalCube<size_t, Layout_CRDB>(pooled_height, pooled_width, iD, iB);

  report_forward_constructor.end(0, 0, 0);
}

/**
 * Forward direction for max pooling
 **/
template <typename DataType>
void MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::forward() {
  report_forward_last_transfer.reset();

  p_output_layer->p_data_cube->reset_cube(-FLT_MAX);

  const LogicalCube<DataType, Layout_CRDB> * const input_data = p_input_layer->p_data_cube;
  LogicalCube<DataType, Layout_CRDB> * const output_data = p_output_layer->p_data_cube;

  const int _pooled_height = pooled_height;
  const int _pooled_width = pooled_width;
  const int _stride = stride;
  const int _kernel_size = kernel_size;
  const int _iR = iR, _iC = iC;

  const DataType * input_data_pdata = input_data->get_logical_matrix(0, 0).p_data;
  DataType * output_data_pdata = output_data->get_logical_matrix(0, 0).p_data;
  size_t * max_index_slice_pdata = max_index->get_logical_matrix(0, 0).p_data;

  const int inc_input = input_data->R*input_data->C;
  const int inc_output = output_data->R*output_data->C;
  const int inc_max = max_index->R*max_index->C;

  for (size_t b_i = 0; b_i < iB; ++b_i) {
    for (size_t d_i = 0; d_i < iD; ++d_i) {

      for (int ph = 0; ph < _pooled_height; ++ph) {
        const int h_start = ph * _stride;
        const int h_end = min(h_start + _kernel_size, _iR);
        for (int pw = 0; pw < _pooled_width; ++pw) {
          const int w_start = pw * _stride;
          const int w_end = min(w_start + _kernel_size, _iC);
          const int pool_index = ph * _pooled_width + pw;
          for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
              const int index = h * _iC + w;

              max_index_slice_pdata[pool_index] = input_data_pdata[index] > output_data_pdata[pool_index] ?
                                              index : max_index_slice_pdata[pool_index];
              output_data_pdata[pool_index] = input_data_pdata[index] > output_data_pdata[pool_index] ? 
                                              input_data_pdata[index] : output_data_pdata[pool_index];
            }
          }
        }
      }

      input_data_pdata += inc_input;
      output_data_pdata += inc_output;
      max_index_slice_pdata += inc_max;
    }
  }

  report_forward_last_transfer.end(1.0*iB*iD*iR*iC*sizeof(DataType), 
          iB*iD*pooled_height*pooled_width*(sizeof(DataType)+sizeof(size_t)), 0);
  report_forward_history.aggregate(report_forward_last_transfer);
}


/**
 * Backward direction for max pooling. (Note: we don't handle the case of max ties)
 **/
template <typename DataType>
void MaxPoolingBridge<DataType, Layout_CRDB, DataType, Layout_CRDB>::backward() {

  report_backward_updateweight_last_transfer.reset();

  p_input_layer->p_gradient_cube->reset_cube();

  const LogicalCube<DataType, Layout_CRDB>* const input_grad = p_input_layer->p_gradient_cube;
  LogicalCube<DataType, Layout_CRDB>* const output_grad = p_output_layer->p_gradient_cube;
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
