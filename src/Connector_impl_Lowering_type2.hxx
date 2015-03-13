//  Connector_impl_Lowering_type2.hxx
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Connector_impl_Lowering_type2_hxx
#define moka_Connector_impl_Lowering_type2_hxx

template<typename DataType, LayoutType InputLayout>
Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE2>::
Connector(const InputLogicalCubeType  * const p_input_cube, const OutputLogicalCubeType * const p_output_cube,
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
  assert(iR == kernel_size);
  assert(iC == kernel_size);
  assert(oR == kernel_size*kernel_size*iB);
  assert(oC == iD);
#endif
  report_constructor.end(0, 0, 0);
}

template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE2>::
lower_model_cube(const InputLogicalCubeType * const p_input_cube, OutputLogicalCubeType * p_output_cube) {

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

  p_output_cube->reset_cube();
  for (size_t i_b = 0; i_b < iB; ++i_b) {
    for (size_t i_d = 0; i_d < iD; ++i_d) {
      const LogicalMatrix<DataType> m = p_input_cube->get_logical_matrix(i_d, i_b);
      p_output_cube->template lower_logical_matrix<LOWERING_TYPE2>(&m, i_b, i_d, kernel_size);
    }
  }

  report_last_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_lowering);
}

template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE2>::
lower_data_cube(const InputLogicalCubeType * const p_input_cube, OutputLogicalCubeType * p_output_cube) {

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

  p_output_cube->reset_cube();
  DataType * output_data = p_output_cube->get_p_data();

  for (size_t i_d = 0; i_d < iD; ++i_d) {
    for (size_t i_b = 0; i_b < iB; ++i_b) {
      const LogicalMatrix<DataType> m = p_input_cube->get_logical_matrix(i_d, i_b);
      Util::_our_memcpy(output_data, m.p_data, sizeof(DataType)*m.n_elements);
      output_data += m.n_elements;
    }
  }

  report_last_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_lowering);
}

template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE2>::
inverse_lower_model_cube(OutputLogicalCubeType * p_output_cube, InputLogicalCubeType * p_input_cube) {

  report_last_inverse_lowering.reset();
  assert(p_input_cube->n_elements == p_output_cube->n_elements);

  DataType * output_data = p_output_cube->get_p_data();
  int i = 0;


  for (size_t r = 0; r < p_input_cube->R; ++r) { // this determines which batch, too
    for (size_t start_c = 0; start_c < p_output_cube->D; ++start_c) {
      for (size_t c = start_c; c < p_input_cube->C; c += p_output_cube->D) {
        output_data[i++] = *p_input_cube->logical_get(r, c, 0, 0);
      }
    }
  }

  report_last_inverse_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_inverse_lowering);
}

template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE2>::
inverse_lower_data_cube(OutputLogicalCubeType * p_output_cube, InputLogicalCubeType * p_input_cube) {

  report_last_inverse_lowering.reset();

  assert(p_input_cube->n_elements == p_output_cube->n_elements);

  DataType * input_data = p_input_cube->get_p_data();
  const DataType * const output_data = p_output_cube->get_p_data();
  const size_t n_elems_single_matrix = p_input_cube->R*p_input_cube->C;

  for (size_t i_b = 0; i_b < p_input_cube->B; ++i_b) {
    for (size_t i_d = 0; i_d < p_input_cube->D; ++i_d) {
      Util::_our_memcpy(input_data, &output_data[i_d*p_output_cube->C + i_b*n_elems_single_matrix],
          sizeof(DataType)*n_elems_single_matrix);
      input_data += n_elems_single_matrix;
    }
  }

  report_last_inverse_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_inverse_lowering);
}

#endif
