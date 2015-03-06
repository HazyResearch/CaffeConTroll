//
//  Connector_impl_Lowering.hxx
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Connector_imple_Lowering_type1_hxx
#define moka_Connector_imple_Lowering_type1_hxx

#include <iostream>

template<typename DataType, LayoutType InputLayout>
Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE1>::
Connector(const InputLogicalCubeType * const p_input_cube, 
  const OutputLogicalCubeType * const p_output_cube, 
  const size_t _kernel_size, const size_t _padding, const size_t _stride,
  DeviceDriver * _p_driver) :
  iR(p_input_cube->R), iC(p_input_cube->C), iD(p_input_cube->D), iB(p_input_cube->B),
  oR(p_output_cube->R), oC(p_output_cube->C), oD(p_output_cube->D), oB(p_output_cube->B),
  kernel_size(_kernel_size), padding(_padding), stride(_stride),
  p_driver(_p_driver) 
{
  report_constructor.reset();
  report_last_lowering.reset();
  report_history.reset();
  report_last_inverse_lowering.reset();
  report_inverse_history.reset();

#ifdef _DO_ASSERT
  assert(oD == 1);
  assert(oB == 1);
  assert(oR == kernel_size * kernel_size * iD);
  assert(oC == ((iR + 2 * padding - kernel_size) / stride + 1) * ((iC + 2 * padding - kernel_size) / stride + 1) * iB);
#endif
  report_constructor.end(0, 0, 0);
}

struct _func_src_to_dst_arg_helper_lower1{
  size_t iD;
  size_t iR;
  size_t iC;
  size_t row_multipler;
  size_t col_multipler;
  size_t kernel_size;
  size_t padding;
  size_t stride;
  size_t oC;
};

// TOFIX TO GPU
__host__ __device__
size_t _func_src_to_dst_conv_lowering(size_t _input_idx, void * curry){
  const _func_src_to_dst_arg_helper_lower1 * const parg
    = reinterpret_cast<_func_src_to_dst_arg_helper_lower1*>(curry);
      // TODO: BETTER ABSTRACTION IS POSSIBLE HERE for the IDX function.
      // Just change to varidic template?

  const size_t input_idx = _input_idx/sizeof(float); // TODO, this uglyness implies that DeviceMemoryPointer needs a type.
  const size_t i_b = input_idx/(parg->iD*parg->iR*parg->iC);
  const size_t i_d = (input_idx/(parg->iR*parg->iC)) % parg->iD;
  const size_t output_row = i_d * parg->row_multipler;
  const size_t output_col = i_b * parg->col_multipler;
  return (output_row*parg->oC + output_col) * sizeof(float);

}
__device__
FUNC_IDX_MAPPING func_src_to_dst_conv_lowering = _func_src_to_dst_conv_lowering;

__host__ __device__
void _func_lowering(void * _dst, void * _src, void * curry){

  const _func_src_to_dst_arg_helper_lower1 * const parg
    = reinterpret_cast<_func_src_to_dst_arg_helper_lower1*>(curry);

  int iR = parg->iR;
  int iC = parg->iC;
  int oC = parg->oC;
  int padding = parg->padding;
  int kernel_size = parg->kernel_size;
  int stride = parg->stride;
  int _kernel_size = parg->kernel_size;

  /*
  int iR = 20;
  int iC = 20;
  int oC = 10;
  int padding = 2;
  int kernel_size = 5;
  int stride = 2;
  int _kernel_size = 5;
  */

  const int height = iR;
  const int width = iC;
  const float * const input_data = (float *) _src;
  float * const output_data = (float *) _dst;

  const int num_height_windows = (height + 2 * padding - kernel_size) / stride + 1; // number of convolution "windows" row-wise
  const int num_width_windows = (width + 2 * padding - kernel_size) / stride + 1; // number of convolution "windows" column-wise

  // i & j keep track of which "window" we're currently calculating. Incremented by 1 each time.
  // src_row_base and src_col_base indicate the starting indices for the window. Incremented by stride each time.
  // dst_col increases every time we calculate a new windo. Incremented by 1 each time.
  for (int src_row_base = -padding, i = 0, dst_col = 0; i < num_height_windows; src_row_base += stride, ++i) {
    for (int src_col_base = -padding, j = 0; j < num_width_windows; src_col_base += stride, ++dst_col, ++j) {
      // src_row and src_col start at src_row_base and src_col_base, respectively, and iterate a total of kernel_size times. Incremented by 1 each time.
      // dst_row_i starts at dst_row_base. Incremented by kernel_size each time.
      // dst_row starts at dst_row_i. Incremented by 1 each time.
      for (int src_row = src_row_base, dst_row_i = 0; src_row < _kernel_size + src_row_base; ++src_row, dst_row_i += _kernel_size) {
        for (int src_col = src_col_base, dst_row = dst_row_i; src_col < _kernel_size + src_col_base; ++src_col, ++dst_row) {
          const size_t dst = dst_col + dst_row*oC;
          if (src_row < 0 || src_row >= width || src_col < 0 || src_col >= height) {
            output_data[dst] = 0;
          } else {
            output_data[dst] = input_data[src_row*width + src_col];
          }
        }
      }
    }
  }
}
__device__ 
FUNC_MM_MAPPING func_lowering = _func_lowering;

/**

  auto func_src_to_dst = [=](size_t _input_idx){
    const size_t input_idx = _input_idx/sizeof(DataType); // TODO, this uglyness implies that DeviceMemoryPointer needs a type.
    const size_t i_b = input_idx/(iD*iR*iC);
    const size_t i_d = (input_idx/(iR*iC)) % iD;
    const size_t output_row = i_d * kernel_size * kernel_size;
    const size_t output_col = i_b * ((iR + 2 * padding - kernel_size) / stride + 1) * ((iC + 2 * padding - kernel_size) / stride + 1);
    return (output_row*oC + output_col) * sizeof(DataType);
  };

  auto func_lowering = [=](void * _dst, const void * _src){
      const int height = iR;
      const int width = iC;
      const DataType * const input_data = (DataType *) _src;
      DataType * const output_data = (DataType *) _dst;

      int _kernel_size = kernel_size;

      const int num_height_windows = (height + 2 * padding - kernel_size) / stride + 1; // number of convolution "windows" row-wise
      const int num_width_windows = (width + 2 * padding - kernel_size) / stride + 1; // number of convolution "windows" column-wise

      // i & j keep track of which "window" we're currently calculating. Incremented by 1 each time.
      // src_row_base and src_col_base indicate the starting indices for the window. Incremented by stride each time.
      // dst_col increases every time we calculate a new windo. Incremented by 1 each time.
      for (int src_row_base = -padding, i = 0, dst_col = 0; i < num_height_windows; src_row_base += stride, ++i) {
        for (int src_col_base = -padding, j = 0; j < num_width_windows; src_col_base += stride, ++dst_col, ++j) {
          // src_row and src_col start at src_row_base and src_col_base, respectively, and iterate a total of kernel_size times. Incremented by 1 each time.
          // dst_row_i starts at dst_row_base. Incremented by kernel_size each time.
          // dst_row starts at dst_row_i. Incremented by 1 each time.
          for (int src_row = src_row_base, dst_row_i = 0; src_row < _kernel_size + src_row_base; ++src_row, dst_row_i += kernel_size) {
            for (int src_col = src_col_base, dst_row = dst_row_i; src_col < _kernel_size + src_col_base; ++src_col, ++dst_row) {
              const size_t dst = dst_col + dst_row*oC;
              if (src_row < 0 || src_row >= width || src_col < 0 || src_col >= height) {
                output_data[dst] = 0;
              } else {
                output_data[dst] = input_data[src_row*width + src_col];
              }
            }
          }
        }
      }
    };
*/

struct _func_src_to_dst_arg_helper_ilower1{
  size_t iD;
  size_t iR;
  size_t iC;
  size_t iB;
  size_t kernel_size;
  size_t padding;
  size_t stride;
  size_t oC;
  size_t data_output_width;
  size_t data_output_height;
};

// TOFIX TO GPU
__host__ __device__
size_t _func_src_to_dst_conv_ilowering(size_t _input_idx, void * curry){
  const _func_src_to_dst_arg_helper_ilower1 * const parg
    = reinterpret_cast<_func_src_to_dst_arg_helper_ilower1*>(curry);
      // TODO: BETTER ABSTRACTION IS POSSIBLE HERE for the IDX function.
      // Just change to varidic template?

  const size_t input_idx = _input_idx/sizeof(float); // TODO, this uglyness implies that DeviceMemoryPointer needs a type.
  const size_t i_b = input_idx/(parg->iD*parg->iR*parg->iC);
  const size_t i_d = (input_idx/(parg->iR*parg->iC)) % parg->iD;
  return (parg->data_output_width * parg->data_output_height * 
    (i_b + i_d * parg->iB*parg->kernel_size*parg->kernel_size))*sizeof(float);

}
__device__
FUNC_IDX_MAPPING func_src_to_dst_conv_ilowering = _func_src_to_dst_conv_ilowering;

__host__ __device__
void _func_ilowering(void * _dst, void * _src, void * curry){

  const _func_src_to_dst_arg_helper_ilower1 * const parg
    = reinterpret_cast<_func_src_to_dst_arg_helper_ilower1*>(curry);

  float * const input_data = (float *) _src;
  const float * const output_data = (float *) _dst;

  for (size_t kr = 0; kr < parg->kernel_size; ++kr) {
    for (size_t kc = 0; kc < parg->kernel_size; ++kc) {

      size_t out_index = parg->data_output_width * parg->data_output_height *
        (kr * parg->iB*parg->kernel_size + kc * parg->iB );

      for (size_t cr = 0; cr < parg->stride * parg->data_output_width; cr += parg->stride) {
        const int input_row_index = cr + kr - parg->padding;

        for (size_t cc = 0; cc < parg->stride * parg->data_output_height; cc += parg->stride) {
          const int input_col_index = cc + kc - parg->padding;

          // (cr + kr - padding, cc + kc - padding) represents the index into
          // the input gradient cube. If we aren't within [0, iR) and [0, iC)
          // then we shouldn't update, because we are in the padded area
          if (input_row_index >= 0 &&
              input_row_index < parg->iR  &&
              input_col_index >= 0 &&
              input_col_index < parg->iC) {
            input_data[input_row_index*parg->iC + input_col_index] += output_data[out_index];
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
__device__
FUNC_MM_MAPPING func_ilowering = _func_ilowering;


/*
  auto func_src_to_dst = [=](size_t _input_idx){
    const size_t input_idx = _input_idx/sizeof(float); // TODO, this uglyness implies that DeviceMemoryPointer needs a type.
    const size_t i_b = input_idx/(iD*parg->iR*iC);
    const size_t i_d = (input_idx/(iR*iC)) % iD;
    return (data_output_width * data_output_height * 
      (i_b + i_d * iB*kernel_size*kernel_size))*sizeof(float);
  };

  auto func_inverse_lowering = [=](void * _dst, void * _src){

    DataType * const input_data = (DataType *) _src;
    const DataType * const output_data = (DataType *) _dst;

    for (size_t kr = 0; kr < kernel_size; ++kr) {
      for (size_t kc = 0; kc < kernel_size; ++kc) {

        size_t out_index = data_output_width * data_output_height *
          (kr * iB*kernel_size + kc * iB );

        for (size_t cr = 0; cr < stride * data_output_width; cr += stride) {
          const int input_row_index = cr + kr - padding;

          for (size_t cc = 0; cc < stride * data_output_height; cc += stride) {
            const int input_col_index = cc + kc - padding;

            // (cr + kr - padding, cc + kc - padding) represents the index into
            // the input gradient cube. If we aren't within [0, iR) and [0, iC)
            // then we shouldn't update, because we are in the padded area
            if (input_row_index >= 0 &&
                input_row_index < iR  &&
                input_col_index >= 0 &&
                input_col_index < iC) {
              input_data[input_row_index*iC + input_col_index] += output_data[out_index];
            }
            // increment out_index regardless, a single cell from the output gradient cube
            // can only make a single contribution to the input gradient cube (Remember: this
            // is the *lowered* output gradient cube!)
            ++out_index;
          }
        }
      }
    }
  };
  */


template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE1>::
lower_cube(const InputLogicalCubeType * const p_input_cube, OutputLogicalCubeType * p_output_cube) {

  report_last_lowering.reset();

#ifdef _DO_ASSERT
  assert(p_input_cube->R == iR);
  assert(p_input_cube->C == iC);
  assert(p_input_cube->D == iD);
  assert(p_input_cube->B == iB);
  assert(p_output_cube->R == oR);
  assert(p_output_cube->C == oC);
  assert(p_output_cube->D == oD);
  assert(p_output_cube->B == oB);
#endif

  DeviceMemoryPointer * input = p_input_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = p_output_cube->get_device_pointer(p_driver);

  _func_src_to_dst_arg_helper_lower1 arg1;
  arg1.iD = iD;
  arg1.iR = iR;
  arg1.iC = iC;
  arg1.row_multipler = kernel_size * kernel_size;
  arg1.col_multipler = ((iR + 2 * padding - kernel_size) / stride + 1) * ((iC + 2 * padding - kernel_size) / stride + 1);;
  arg1.kernel_size = kernel_size;
  arg1.padding = padding;
  arg1.stride = stride;
  arg1.oC = oC;

  DeviceMemoryPointer * parg1 = p_driver->get_device_pointer((void*)&arg1, 
    sizeof(_func_src_to_dst_arg_helper_lower1));

  DeviceMemoryPointer * parg2 = p_driver->get_device_pointer((void*)&arg1, 
    sizeof(_func_src_to_dst_arg_helper_lower1));

  p_driver->parallel_map(output, input, iR*iC*sizeof(DataType), 
    &func_src_to_dst_conv_lowering, parg1, &func_lowering, parg2);

  report_last_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_lowering);
}

template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE1>::
inverse_lower_cube(OutputLogicalCubeType * p_output_cube, InputLogicalCubeType * p_input_cube) {

  report_last_inverse_lowering.reset();

#ifdef _DO_ASSERT
  assert(p_input_cube->R == iR);
  assert(p_input_cube->C == iC);
  assert(p_input_cube->D == iD);
  assert(p_input_cube->B == iB);
  assert(p_output_cube->R == oR);
  assert(p_output_cube->C == oC);
  assert(p_output_cube->D == oD);
  assert(p_output_cube->B == oB);
#endif

  DeviceMemoryPointer * input = p_input_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = p_output_cube->get_device_pointer(p_driver);

  p_driver->sconstant_initialize(input, DataType(0.0));

  const size_t data_output_width = (iR + 2 * padding - kernel_size) / stride + 1;  // the number of rows in the output gradient cube
  const size_t data_output_height = (iC + 2 * padding - kernel_size) / stride + 1; // the number of cols in the output gradient cube

  _func_src_to_dst_arg_helper_ilower1 arg1;

  arg1.iD = iD;
  arg1.iR = iR;
  arg1.iC = iC;
  arg1.iB = iB;
  arg1.kernel_size = kernel_size;
  arg1.padding = padding;
  arg1.stride = stride;
  arg1.oC = oC;
  arg1.data_output_width = data_output_width;
  arg1.data_output_height = data_output_height;

  DeviceMemoryPointer * parg1 = p_driver->get_device_pointer((void*)&arg1, 
    sizeof(_func_src_to_dst_arg_helper_ilower1));

  DeviceMemoryPointer * parg2 = p_driver->get_device_pointer((void*)&arg1, 
    sizeof(_func_src_to_dst_arg_helper_ilower1));

  p_driver->parallel_map(output, input, iR*iC*sizeof(DataType), 
    &func_src_to_dst_conv_ilowering, parg1, &func_ilowering, parg2);

  report_last_inverse_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_inverse_lowering);
}



#endif
