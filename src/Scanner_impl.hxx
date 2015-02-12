//
//  Scanner_impl.hxx
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Scanner_impl_hxx
#define moka_Scanner_impl_hxx

template
<typename DataType, LayoutType Layout>
Scanner<DataType, Layout, FUNC_TANH>::Scanner(const LogicalCubeType * const p_cube){
    report_constructor.reset();
    report_last_apply.reset();
    report_history.reset();

    report_constructor.end(0, 0, 0);
};


template
<typename DataType, LayoutType Layout>
void Scanner<DataType, Layout, FUNC_TANH>::apply(LogicalCubeType * const p_cube){
    report_last_apply.reset();

    size_t i;
    for(i=0;i<p_cube->n_elements;i++){
        // According to Intel's Manual, this should be automatically vectorized
        //    TODO: But I coud make EXP for TANH faster if necessary -- but this layer is so much faster than GEMM
        p_cube->p_data[i] = tanh(p_cube->p_data[i]);
    }

    report_last_apply.end(1.0*p_cube->n_elements*sizeof(DataType), 1.0*p_cube->n_elements*sizeof(DataType), 1.0*p_cube->n_elements);
    report_history.aggregate(report_last_apply);

};


#endif
