//
//  Kernel.h
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "Cube.h"
#include "Report.h"

#ifndef moka_Kernel_h
#define moka_Kernel_h

enum KernelType{
    Kernel_GEMM_OpenBlas = 0,
    Kernel_GEMM_Magma = 1,
    Kernel_ELEMENTWISEMUL_CPU = 2
};

enum KernelConfig{
    KernelConfig_NONE = 0,
    KernelConfig_GEMM_NOTRANS_NOTRANS = 1,
    KernelConfig_GEMM_NOTRANS_TRANS = 2,
    KernelConfig_GEMM_TRANS_NOTRANS = 3,
    KernelConfig_TANHGRAD_ON_INPUT1 = 4
};

template
<typename Input1DataType, LayoutType Input1Layout,
typename Input2DataType, LayoutType Input2Layout,
typename OutputDataType, LayoutType OutputLayout,
KernelType KERNELTYPE, KernelConfig KERNELCONFIG>
class Kernel{
public:
    
    typedef Cube<Input1DataType, Input1Layout> Input1CubeType;
    typedef Cube<Input2DataType, Input2Layout> Input2CubeType;
    typedef Cube<OutputDataType, OutputLayout> OutputCubeType;
    
    const size_t i1R, i1C, i1D, i1B; /*< Size of the input Cube 1 */
    const size_t i2R, i2C, i2D, i2B; /*< Size of the input Cube 2 */
    const size_t oR, oC, oD, oB; /*< Size of the output Cube */
    
    Report report_constructor; /*< Performance reporter for constructor function. */
    Report report_last_transfer; /*< Performance reporter for the last run of transfer() function. */
    Report report_history; /*< Performance reporter for all transfer() functions aggregated. */
    
    /**
     * Similar to Connector()'s constructor.
     **/
    Kernel(const Input1CubeType * const p_input1_cube,
           const Input2CubeType * const p_input2_cube,
           const OutputCubeType * const p_output_cube){
        std::cerr << "ERROR: Using a kernel with unsupported Layout or DataType." << std::endl;
        assert(false);
    }
    
    void compute(const Input1CubeType * const p_input1_cube,
                  const Input2CubeType * const p_input2_cube,
                  OutputCubeType * const p_output_cube){
        std::cerr << "ERROR: Using a kernel with unsupported Layout or DataType." << std::endl;
        assert(false);
    }
    
};


/******
 * Specializations
 */
template <typename DataType, KernelConfig KERNELCONFIG>
class Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_GEMM_OpenBlas, KERNELCONFIG>{
public:
    
    typedef Cube<DataType, Layout_CRDB> Input1CubeType;
    typedef Cube<DataType, Layout_CRDB> Input2CubeType;
    typedef Cube<DataType, Layout_CRDB> OutputCubeType;
    
    float alpha;
    float beta;
    
    char transA;
    char transB;
    
    const size_t i1R, i1C, i1D, i1B;
    const size_t i2R, i2C, i2D, i2B;
    const size_t oR, oC, oD, oB;
    
    Report report_constructor;
    Report report_last_transfer;
    Report report_history;

    Kernel(const Input1CubeType * const p_input1_cube, const Input2CubeType * const p_input2_cube,
           const OutputCubeType * const p_output_cube);
    
    void compute(const Input1CubeType * const p_input1_cube, const Input2CubeType * const p_input2_cube,
                  OutputCubeType * const p_output_cube);
    
};

template <typename DataType, KernelConfig KERNELCONFIG>
class Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KERNELCONFIG>{
public:
    
    typedef Cube<DataType, Layout_CRDB> Input1CubeType;
    typedef Cube<DataType, Layout_CRDB> Input2CubeType;
    typedef Cube<DataType, Layout_CRDB> OutputCubeType;
    
    char transA;
    char transB;
    
    const size_t i1n_elements;
    const size_t i2n_elements;
    const size_t on_elements;
    
    Report report_constructor;
    Report report_last_transfer;
    Report report_history;
    
    Kernel(const Input1CubeType * const p_input1_cube, const Input2CubeType * const p_input2_cube,
           const OutputCubeType * const p_output_cube);
    
    void compute(const Input1CubeType * const p_input1_cube, const Input2CubeType * const p_input2_cube, OutputCubeType * const p_output_cube);
    
};


#include "Kernel_impl.hxx"

#endif
