//
//  Kernel.h
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Kernel_h
#define moka_Kernel_h

#include "sched/DeviceDriver.h"

#include "LogicalCube.h"
#include "Report.h"

enum KernelType {
    Kernel_GEMM_OpenBlas = 0,
    Kernel_GEMM_Magma = 1,
    Kernel_ELEMENTWISEMUL_CPU = 2
};

enum KernelConfig {
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
class Kernel {
public:

    typedef LogicalCube<Input1DataType, Input1Layout> Input1LogicalCubeType;
    typedef LogicalCube<Input2DataType, Input2Layout> Input2LogicalCubeType;
    typedef LogicalCube<OutputDataType, OutputLayout> OutputLogicalCubeType;

    const size_t i1R, i1C, i1D, i1B; /*< Size of the input LogicalCube 1 */
    const size_t i2R, i2C, i2D, i2B; /*< Size of the input LogicalCube 2 */
    const size_t oR, oC, oD, oB; /*< Size of the output LogicalCube */

    Report report_constructor; /*< Performance reporter for constructor function. */
    Report report_last_lowering; /*< Performance reporter for the last run of transfer() function. */
    Report report_history; /*< Performance reporter for all transfer() functions aggregated. */

    DeviceDriver * p_driver;

    /**
     * Similar to Connector()'s constructor.
     **/
    Kernel(const Input1LogicalCubeType * const p_input1_cube,
           const Input2LogicalCubeType * const p_input2_cube,
           const OutputLogicalCubeType * const p_output_cube,
           DeviceDriver * _p_driver){
        std::cerr << "ERROR: Using a kernel with unsupported Layout or DataType." << std::endl;
        assert(false);
    }

    void compute(const Input1LogicalCubeType * const p_input1_cube,
                  const Input2LogicalCubeType * const p_input2_cube,
                  OutputLogicalCubeType * const p_output_cube){
        std::cerr << "ERROR: Using a kernel with unsupported Layout or DataType." << std::endl;
        assert(false);
    }

};

/******
 * Specializations
 */
template <typename DataType, KernelConfig KERNELCONFIG>
class Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_GEMM_OpenBlas, KERNELCONFIG> {
public:

    typedef LogicalCube<DataType, Layout_CRDB> Input1LogicalCubeType;
    typedef LogicalCube<DataType, Layout_CRDB> Input2LogicalCubeType;
    typedef LogicalCube<DataType, Layout_CRDB> OutputLogicalCubeType;

    char transA;
    char transB;

    const size_t i1R, i1C, i1D, i1B;
    const size_t i2R, i2C, i2D, i2B;
    const size_t oR, oC, oD, oB;

    float alpha;
    float beta;

    Report report_constructor;
    Report report_last_lowering;
    Report report_history;

    DeviceDriver * p_driver;

    Kernel(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
           const OutputLogicalCubeType * const p_output_cube, DeviceDriver * _p_driver);

    void compute(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
                  OutputLogicalCubeType * const p_output_cube);

};

template <typename DataType, KernelConfig KERNELCONFIG>
class Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB, Kernel_ELEMENTWISEMUL_CPU, KERNELCONFIG> {
public:

    typedef LogicalCube<DataType, Layout_CRDB> Input1LogicalCubeType;
    typedef LogicalCube<DataType, Layout_CRDB> Input2LogicalCubeType;
    typedef LogicalCube<DataType, Layout_CRDB> OutputLogicalCubeType;

    char transA;
    char transB;

    const size_t i1n_elements;
    const size_t i2n_elements;
    const size_t on_elements;

    Report report_constructor;
    Report report_last_lowering;
    Report report_history;

    DeviceDriver * p_driver;

    Kernel(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
           const OutputLogicalCubeType * const p_output_cube, DeviceDriver * _p_driver);

    void compute(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube, OutputLogicalCubeType * const p_output_cube);

};


#include "Kernel_impl.hxx"

#endif
