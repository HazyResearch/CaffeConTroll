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
Connector(const InputLogicalCubeType * const p_input_cube, const OutputLogicalCubeType * const p_output_cube,
    const size_t _kernel_size, const size_t _padding, const size_t _stride) :
  iR(p_input_cube->R), iC(p_input_cube->C), iD(p_input_cube->D), iB(p_input_cube->B),
  oR(p_output_cube->R), oC(p_output_cube->C), oD(p_output_cube->D), oB(p_output_cube->B),
  kernel_size(_kernel_size), padding(_padding), stride(_stride)
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

  for (size_t i_b = 0; i_b < iB; ++i_b) {
    for (size_t i_d = 0; i_d < iD; ++i_d) {
      const LogicalMatrix<DataType> m = p_input_cube->get_logical_matrix(i_d, i_b);
      // TODO: instead of evaluating this if check iB*iD times,
      // we should use a function pointer instead
      if (stride == 1 && padding == 0) {
        p_output_cube->template lower_logical_matrix<LOWERING_TYPE1>(&m, i_b, i_d, kernel_size);
      } else {
        p_output_cube->template lower_logical_matrix<LOWERING_TYPE1>(&m, i_b, i_d, kernel_size,
            stride, padding);
      }
    }
  }

  report_last_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_lowering);
}

template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE1>::
inverse_lower_cube(OutputLogicalCubeType * p_output_cube, InputLogicalCubeType * p_input_cube) {

  report_last_inverse_lowering.reset();

#ifdef _DO_WARNING
  cerr << "WARNING: " << "You are using the most general version of the lowering function. " << "This might be slow!" << endl;
#endif

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

  p_input_cube->reset_cube();

  size_t out_index = 0;
  const DataType * const output_data = p_output_cube->get_p_data();

  const size_t data_output_width = (iR + 2 * padding - kernel_size) / stride + 1;  // the number of rows in the output gradient cube
  const size_t data_output_height = (iC + 2 * padding - kernel_size) / stride + 1; // the number of cols in the output gradient cube

  // First, we iterate over K * K * iD , which is the number of rows in the output gradient
  // cube. (Remember: the output gradient cube has dimensions K * K * iD x oR * oC * iB x 1 x 1,
  // where oR and oC do NOT refer the variables above. TODO: We REALLY need to standardize this!)
  for (size_t kd = 0; kd < iD; ++kd) {
    for (size_t kr = 0; kr < kernel_size; ++kr) {
      for (size_t kc = 0; kc < kernel_size; ++kc) {

        // Then, we iterate over oR * oC * iB, the number of columns in the output gradient
        for (size_t ib = 0; ib < iB; ++ib) {
          // cr and cc represent the row index and column index of the convolutional "window"
          // in the input gradient cube, which means that they must be incremented by stride
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
                *p_input_cube->logical_get(input_row_index, input_col_index, kd, ib) += output_data[out_index];
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
  }

  report_last_inverse_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_inverse_lowering);
}

#endif
