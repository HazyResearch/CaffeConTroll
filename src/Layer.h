//
//  Layer.h
//  moka
//
//  Created by Ce Zhang on 1/13/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#include "Cube.h"
#include "Connector.h"
#include "Kernel.h"
#include "Report.h"

#ifndef moka_Layer_h
#define moka_Layer_h

/**
 * One Layer contains three things:
 *   - [DATA] A Cube that is the Data in this Layer;
 *   - [MODEL] A Cube that is the Model which we will use for the Forward Pass;
 *   - [GRADIENT] A Cube that is the Gradient of Data which we will use for Backward Pass.
 **/
template
<typename DataType, LayoutType DataLayout>
class Layer{
public:
    
    typedef Cube<DataType, DataLayout> DataCubeType;
    typedef Cube<DataType, DataLayout> ModelCubeType;
    typedef Cube<DataType, DataLayout> GradientCubeType;
    
    const size_t dR, dC, dD, dB;
    const size_t mR, mC, mD, mB;
    const size_t gR, gC, gD, gB;
    
    DataCubeType * const p_data_cube;
    ModelCubeType * const p_model_cube;
    GradientCubeType * const p_gradient_cube;
    
    Layer(DataCubeType * const _p_data_cube,
          ModelCubeType * const _p_model_cube,
          GradientCubeType * const _p_gradient_cube) :
        p_data_cube(_p_data_cube),
        p_model_cube(_p_model_cube),
        p_gradient_cube(_p_gradient_cube),
        dR(_p_data_cube->R), dC(_p_data_cube->C), dD(_p_data_cube->D), dB(_p_data_cube->B),
        mR(_p_model_cube->R), mC(_p_model_cube->C), mD(_p_model_cube->D), mB(_p_model_cube->B),
        gR(_p_gradient_cube->R), gC(_p_gradient_cube->C), gD(_p_gradient_cube->D), gB(_p_gradient_cube->B)
    {
#ifdef _DO_ASSERT
        assert(dR==gR); assert(dC==gC); assert(dD==gD); assert(dB==gB);
#endif
    }
};

#endif




