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
Connector<DataType, InputLayout, DataType, Layout_CRDB, Connector_Lowering_TYPE1>::
Connector(const InputLogicalCubeType  * const p_input_cube, const OutputLogicalCubeType * const p_output_cube,
          const void * const _p_config) :
  iR(p_input_cube->R), iC(p_input_cube->C), iD(p_input_cube->D), iB(p_input_cube->B),
  oR(p_output_cube->R), oC(p_output_cube->C), oD(p_output_cube->D), oB(p_output_cube->B),
  p_config((LoweringConfig*)_p_config)
{

    report_constructor.reset();
    report_last_transfer.reset();
    report_history.reset();
    report_last_inverse_transfer.reset();
    report_inverse_history.reset();

#ifdef _DO_ASSERT
    const size_t & ksize = p_config->kernel_size;
    assert(oD==1);
    assert(oB==1);
    assert(oR==ksize*ksize*iD);
    assert(oC==(iR-ksize+1)*(iC-ksize+1)*iB);
#endif
    report_constructor.end(0, 0, 0);
}

template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, Connector_Lowering_TYPE1>::
transfer(const InputLogicalCubeType * const p_input_cube, OutputLogicalCubeType * p_output_cube){

    report_last_transfer.reset();

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

    #pragma unroll
    for(size_t kd = 0; kd < iD; kd++){
      #pragma unroll
      for(size_t ib = 0; ib < iB; ib++){
        const LogicalMatrix<DataType> m = p_input_cube->get_logical_matrix(kd, ib);
        std::cout << "LOGICAL MATRIX PHYSICAL PRINT: " << std::endl;
        m.physical_print();
        p_output_cube->append_logical_matrix(&m, ib, kd, p_config->kernel_size, p_config->stride);
      }
    }

    report_last_transfer.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
    report_history.aggregate(report_last_transfer);
}

template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, Connector_Lowering_TYPE1>::
old_transfer(const InputLogicalCubeType * const p_input_cube, OutputLogicalCubeType * p_output_cube){

    report_last_transfer.reset();

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

    const size_t & ksize = p_config->kernel_size;
    size_t outr = 0, outc = 0;

    #pragma unroll
    for(size_t kd = 0; kd<iD; kd++){
        #pragma unroll
        for(size_t kr = 0; kr<ksize; kr++){
            #pragma unroll
            for(size_t kc = 0; kc<ksize; kc++){

                outc = 0;
                #pragma unroll
                for(size_t ib = 0; ib < iB; ib++){
                    #pragma unroll
                    for(size_t cr = 0; cr < iR - ksize + 1; cr++){
                    //for(size_t cr = 0; cr < ksize; cr++){
                        #pragma unroll
                        for(size_t cc = 0; cc < iC - ksize + 1; cc++){
                        //for(size_t cc = 0; cc < ksize; cc++){
                            *p_output_cube->logical_get(outr, outc, 0, 0) = *p_input_cube->logical_get(cr+kr, cc+kc, kd, ib);
                            outc ++;
                        }
                    }
                }
                outr ++;
            }
        }
    }

    report_last_transfer.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
    report_history.aggregate(report_last_transfer);
}


template<typename DataType, LayoutType InputLayout>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, Connector_Lowering_TYPE1>::
inverse_transfer(OutputLogicalCubeType * p_output_cube, InputLogicalCubeType * p_input_cube){

    report_last_inverse_transfer.reset();

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

    for(size_t c=0;c<iC;c++){
        for(size_t r=0;r<iR;r++){
            for(size_t d=0;d<iD;d++){
                for(size_t b=0;b<iB;b++){
                    *p_input_cube->logical_get(r, c, d, b) = 0;
                }
            }
        }
    }

    const size_t & ksize = p_config->kernel_size;
    size_t outr = 0, outc = 0;

    #pragma unroll
    for(size_t kd=0;kd<iD;kd++){
        #pragma unroll
        for(size_t kr=0;kr<ksize;kr++){
            #pragma unroll
            for(size_t kc=0;kc<ksize;kc++){

                outc = 0;
                #pragma unroll
                for(size_t ib=0;ib<iB;ib++){
                    #pragma unroll
                    for(size_t cr=0;cr<iR-ksize+1;cr++){
                        #pragma unroll
                        for(size_t cc=0;cc<iC-ksize+1;cc++){
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

    report_last_inverse_transfer.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
    report_history.aggregate(report_last_inverse_transfer);
}


#endif
