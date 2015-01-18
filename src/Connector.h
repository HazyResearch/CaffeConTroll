//
//  Connector.h
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "LogicalCube.h"
#include "Report.h"
#include "LogicalMatrix.h"

#ifndef moka_Connector_h
#define moka_Connector_h

enum ConnectorType{
    Connector_Lowering_TYPE1 = 0, // we definitely need better names -- but these three are the three types of lowering algorithms
    Connector_Lowering_TYPE2 = 1,
    Connector_Lowering_TYPE3 = 2
};

struct LoweringConfig {
    size_t kernel_size;
    size_t stride;
};

template
<typename InputDataType, LayoutType InputLayout,
 typename OutputDataType, LayoutType OutputLayout,
 ConnectorType CONNECTOR>
class Connector {
public:

    typedef LogicalCube<InputDataType, InputLayout> InputLogicalCubeType;
    typedef LogicalCube<OutputDataType, OutputLayout> OutputLogicalCubeType;

    const size_t iR, iC, iD, iB; /*< Size of the input LogicalCube */
    const size_t oR, oC, oD, oB; /*< Size of the output LogicalCube */

    Report report_constructor; /*< Performance reporter for constructor function. */
    Report report_last_transfer; /*< Performance reporter for the last run of transfer() function. */
    Report report_history; /*< Performance reporter for all transfer() functions aggregated. */

    Report report_last_inverse_transfer; /*< Performance reporter for the last run of inverse_transfer() function. */
    Report report_inverse_history; /*< Performance reporter for all inverse_transfer() functions aggregated. */

    /**
     * The constructor of a connector allocates necessary memory for
     * transformation (does not include the memory of input/output cube)
     *
     * The p_input_cube and p_output_cube input in the constructor is just
     * used for getting the dimensional information of input/output, which
     * is assumed to be constant in later transfer(). No transfer() will be
     * done in the constructor, hinted by that the output LogicalCube is const.
     *
     **/
    Connector(const InputLogicalCubeType * const p_input_cube, const OutputLogicalCubeType * const p_output_cube,
              const void * const p_config){
        std::cerr << "ERROR: Using Connector with the type that is not specialized (implemented)." << std::endl;
        assert(false);
    }

    /**
     * The transfer function takes as input p_input_cube and transfer to
     * p_output_cube.
     *
     **/
    void transfer(InputLogicalCubeType * p_input_cube, OutputLogicalCubeType * p_output_cube){
        std::cerr << "ERROR: Using Connector with the type that is not specialized (implemented)." << std::endl;
        assert(false);
    }

    /**
     * The inverse transfer function that takes as input p_output_cube, and output
     * p_input_cube.
     **/
    void inverse_transfer(OutputLogicalCubeType * p_output_cube, InputLogicalCubeType * p_input_cube){
        std::cerr << "ERROR: Using Connector with the type that is not specialized (implemented)." << std::endl;
        assert(false);
    }

};

/**
 * Specialization for the Lowering Connector (Connector_Lowering_R1C1)
 * where InputDataType == OutputDataType, OutputLayout = Layout_CRDB.
 * No informaton about InputLayout is used, so this is the most general
 * version, but might be slow.
 *
 */
template
<typename DataType, LayoutType InputLayout>
class Connector<DataType, InputLayout, DataType, Layout_CRDB, Connector_Lowering_TYPE1>{
public:

    typedef LogicalCube<DataType, InputLayout> InputLogicalCubeType;
    typedef LogicalCube<DataType, Layout_CRDB> OutputLogicalCubeType;

    const size_t iR, iC, iD, iB;
    const size_t oR, oC, oD, oB;

    Report report_constructor;
    Report report_last_transfer;
    Report report_history;

    Report report_last_inverse_transfer;
    Report report_inverse_history;

    const LoweringConfig * const p_config;

    Connector(const InputLogicalCubeType  * const p_input_cube, const OutputLogicalCubeType * const p_output_cube,
              const void * const _p_config);

    void transfer(const InputLogicalCubeType * const p_input_cube, OutputLogicalCubeType * p_output_cube);
    void old_transfer(const InputLogicalCubeType * const p_input_cube, OutputLogicalCubeType * p_output_cube);

    void inverse_transfer(OutputLogicalCubeType * p_output_cube, InputLogicalCubeType * p_input_cube);

};


template
<typename DataType, LayoutType InputLayout>
class Connector<DataType, InputLayout, DataType, Layout_CRDB, Connector_Lowering_TYPE2>{
public:

    typedef LogicalCube<DataType, InputLayout> InputLogicalCubeType;
    typedef LogicalCube<DataType, Layout_CRDB> OutputLogicalCubeType;

    const size_t iR, iC, iD, iB;
    const size_t oR, oC, oD, oB;

    Report report_constructor;
    Report report_last_transfer;
    Report report_history;

    Report report_last_inverse_transfer;
    Report report_inverse_history;

    const LoweringConfig * const p_config;

    Connector(const InputLogicalCubeType  * const p_input_cube, const OutputLogicalCubeType * const p_output_cube,
              const void * const _p_config);

    void transfer(const InputLogicalCubeType * const p_input_cube, OutputLogicalCubeType * p_output_cube);

    void inverse_transfer(OutputLogicalCubeType * p_output_cube, InputLogicalCubeType * p_input_cube);

};




/*
template
<typename InputDataType, typename OutputDataType>
class Connector<InputDataType, Layout_CRDB, OutputDataType, Layout_CRDB, Connector_Lowering_R1C1>{
public:

    typedef LogicalCube<InputDataType, Layout_CRDB> InputLogicalCubeType;
    typedef LogicalCube<OutputDataType, Layout_CRDB> OutputLogicalCubeType;

    const size_t iR, iC, iD, iB;
    const size_t oR, oC, oD, oB;

    Connector(const InputLogicalCubeType  * const p_input_cube, const OutputLogicalCubeType * const p_output_cube);

    void transfer(InputLogicalCubeType * p_input_cube, OutputLogicalCubeType * p_output_cube);

};
*/


#include "Connector_impl_Lowering_type1.hxx"
#include "Connector_impl_Lowering_type2.hxx"

#endif
