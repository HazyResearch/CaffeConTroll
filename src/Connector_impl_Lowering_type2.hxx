//  Connector_imple_Lowering_type2.hxx
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Connector_imple_Lowering_type2_hxx
#define moka_Connector_imple_Lowering_type2_hxx

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
  assert(oD==1);
  assert(oB==1);
  assert(oR==kernel_size*kernel_size*iD);
  assert(oC==(iR-kernel_size+1)*(iC-kernel_size+1)*iB);
#endif
  report_constructor.end(0, 0, 0);
}

template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE2>::
lower_cube(const InputLogicalCubeType * const p_input_cube, OutputLogicalCubeType * p_output_cube) {

  report_last_lowering.reset();

#ifdef _DO_WARNING
  std::cerr << "WARNING: " << "You are using the most general version of the lowering function. " << "This might be slow!" << std::endl;
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

  for (size_t kd = 0; kd < iD; kd++) {
    for (size_t ib = 0; ib < iB; ib++) {
      const LogicalMatrix<DataType> m = p_input_cube->get_logical_matrix(kd, ib);
      p_output_cube->template lower_logical_matrix<LOWERING_TYPE2>(&m, ib, kd, kernel_size,
          stride);
    }
  }

  report_last_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_lowering);
}



template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE2>::
inverse_lower_cube(OutputLogicalCubeType * p_output_cube, InputLogicalCubeType * p_input_cube) {

  report_last_inverse_lowering.reset();

#ifdef _DO_WARNING
  std::cerr << "WARNING: " << "You are using the most general version of the lowering function. " << "This might be slow!" << std::endl;
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

  for (size_t c=0;c<iC;c++) {
    for (size_t r=0;r<iR;r++) {
      for (size_t d=0;d<iD;d++) {
        for (size_t b=0;b<iB;b++) {
          *p_input_cube->logical_get(r, c, d, b) = 0;
        }
      }
    }
  }

  size_t outr = 0, outc = 0;

  for (size_t kd=0;kd<iD;kd++) {
    for (size_t kr=0;kr<kernel_size;kr++) {
      for (size_t kc=0;kc<kernel_size;kc++) {

        outc = 0;
        for (size_t ib=0;ib<iB;ib++) {
          for (size_t cr=0;cr<iR-kernel_size+1;cr++) {
            for (size_t cc=0;cc<iC-kernel_size+1;cc++) {
              *p_input_cube->logical_get(cr+kr, cc+kc, kd, ib) +=
                *p_output_cube->logical_get(outr, outc, 0, 0);
              outc ++;
            }
          }
        }
        outr ++;
      }
    }
  }

  report_last_inverse_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_inverse_lowering);
}

#endif
