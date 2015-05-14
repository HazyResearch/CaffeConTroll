//
//  Kernel.h
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

// SHADJIS TODO: Currently we use kernels to do things like 
// GEMM, e.g. instantiate a kernel object and call kernel.compute() 
// which calls driver->sgemm(), rather than just directly call 
// driver->sgemm(). Need to evaluate if this abstraction still makes 
// sense, since e.g. now every different gemm size requires a different
// kernel object. This is because each kernel takes as input 
// cubes which define its size.

#ifndef moka_Kernel_h
#define moka_Kernel_h

#include "sched/DeviceDriver_CPU.h"

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
    KernelConfig_TANHGRAD_ON_INPUT1 = 4,

    // Transpose parameters to BLAS can be interpreted
    // in two ways. Consider if we want to do:
    //
    //       C    =   A   *   B
    //      ___      ___     ___
    //    M|   |   M|   |  K|   |
    //     |___|    |___|   |___|
    //       N        K       N
    // 
    // In BLAS terminology, this is C = op(A) * op(B), where
    // op(A) and op(B) are the transformed versions of A and
    // B after applying transpose or conjugate.
    // See: https://software.intel.com/en-us/node/468480
    //
    // Consider if A is stored in memory how we want (i.e. each row 
    // of size K is concatenated), but B is transposed in memory, i.e. 
    // stored in column-major order instead of row major.
    // Then, the number of elements in B is still K*N = N*K, but there are
    // two possibilities:
    // 
    //  1. B is viewed as a K*N matrix stored in column-major order
    //  2. B is viewed as an N*K matrix stored in row-major order
    // 
    // In case 1, we need to set the blas flag to transpose B, and we
    // need to change LDB, but we do not need to change arguments m,n,k
    // to sgemm. In case 2, we also need to set the flag to transpose B,
    // and we need to give blas m,k,n instead of m,n,k as the sizes, but
    // we do not need to change LDB.
    // 
    // Currently this distinction is important because we do:
    //
    //      R_hat = K_hat * D_hat
    //
    // R_hat is o*mmb, K_hat is o*kkd and D_hat is kkd*mmb
    // The new implementation of lowering following equation 4 of 
    // "Formulation of Type 1 Lowering with Padding and Stride" (see CCT_ROOT/docs)
    // transposes how D_hat is stored in memory. However none of the
    // matrix dimensions changed. So previously it was necessary to
    // just multiply K_hat * D_hat, but now it is necessary to 
    // multiply K_hat * D_hat' (transpose), but without changing the
    // dimensions (i.e. case 1 above). Because KernelConfig_GEMM_NOTRANS_TRANS
    // instead implements case 2 above (i.e. assumes the matrix data 
    // and dimensions are transposed) I added KernelConfig_GEMM_NOTRANS_TRANS_NO_DIM_FLIP
    // to transpose and not flip dimensions.
    // I also added KernelConfig_GEMM_NOTRANS_NOTRANS_DIM_FLIP to flip dimensions but
    // not transpose. 
    // This might not be the best way to do it (e.g. use remap instead) but 
    // if BLAS can handle transposes for "free" this might be faster.
    // SHADJIS TODO: Measure overhead of current remap and decide if matrix multiply
    // should be K_hat*D_hat as it is now or switch to D_hat*K_hat.
    KernelConfig_GEMM_NOTRANS_TRANS_NO_DIM_FLIP = 5,
    KernelConfig_GEMM_TRANS_NO_DIM_FLIP_NOTRANS = 6,
    KernelConfig_GEMM_NOTRANS_NOTRANS_DIM_FLIP = 7,
    KernelConfig_GEMM_NOTRANS_DIM_FLIP_NOTRANS = 8
};

template
<typename Input1DataType, LayoutType Input1Layout,
 typename Input2DataType, LayoutType Input2Layout,
 typename OutputDataType, LayoutType OutputLayout,
 KernelType KERNELTYPE, KernelConfig KERNELCONFIG,
 typename DriverClass>
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

    DriverClass * p_driver;

    /**
     * Similar to Connector()'s constructor.
     **/
    Kernel(const Input1LogicalCubeType * const p_input1_cube,
           const Input2LogicalCubeType * const p_input2_cube,
           const OutputLogicalCubeType * const p_output_cube,
           DriverClass * _p_driver):
        i1R(0), i1C(0), i1D(0), i1B(0),
        i2R(0), i2C(0), i2D(0), i2B(0),
        oR(0), oC(0), oD(0), oB(0)
    {
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
template <typename DataType, KernelConfig KERNELCONFIG, typename DriverClass>
class Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB,
      Kernel_GEMM_OpenBlas, KERNELCONFIG, DriverClass> {
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

    DriverClass * p_driver;

    Kernel(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
           const OutputLogicalCubeType * const p_output_cube, DriverClass * _p_driver);

    void compute(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
                  OutputLogicalCubeType * const p_output_cube);

};

template <typename DataType, KernelConfig KERNELCONFIG, typename DriverClass>
class Kernel<DataType, Layout_CRDB, DataType, Layout_CRDB, DataType, Layout_CRDB,
      Kernel_ELEMENTWISEMUL_CPU, KERNELCONFIG, DriverClass> {
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

    DriverClass * p_driver;

    Kernel(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
           const OutputLogicalCubeType * const p_output_cube, DriverClass * _p_driver);

    void compute(const Input1LogicalCubeType * const p_input1_cube, const Input2LogicalCubeType * const p_input2_cube,
        OutputLogicalCubeType * const p_output_cube);

};

#include "Kernel_impl.hxx"

#endif
