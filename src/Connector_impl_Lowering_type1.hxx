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
//#include "kernels/lowering.h"

template<typename DataType, LayoutType InputLayout, typename DriverClass>
Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE1, DriverClass>::
Connector(const InputLogicalCubeType * const p_input_cube,
  const OutputLogicalCubeType * const p_output_cube,
  const size_t _kernel_size, const size_t _padding, const size_t _stride,
  DriverClass * _p_driver) :
  iR(p_input_cube->R), iC(p_input_cube->C), iD(p_input_cube->D), iB(p_input_cube->B),
  oR(p_output_cube->R), oC(p_output_cube->C), oD(p_output_cube->D), oB(p_output_cube->B),
  kernel_size(_kernel_size), padding(_padding), stride(_stride),
  p_driver(_p_driver)
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


template<typename DataType, LayoutType InputLayout, typename DriverClass>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE1, DriverClass>::
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

  DeviceMemoryPointer * input = p_input_cube->get_device_pointer(p_driver);
  DeviceMemoryPointer * output = p_output_cube->get_device_pointer(p_driver);

  PMapHelper args;
  args.dR = p_output_cube->R; args.dC = p_output_cube->C; args.dD = p_output_cube->D; args.dB = p_output_cube->B;
  args.sR = p_input_cube->R; args.sC = p_input_cube->C; args.sD = p_input_cube->D; args.sB = p_input_cube->B;
  args.dBR = args.dR; args.dBC = args.dC;
  args.sBR = min((size_t)32, args.sR); args.sBC = min((size_t)32, args.sC);
  args.kR = kernel_size; args.kC = kernel_size; args.kD = p_input_cube->D; args.kB = 1;
  args.stride = stride;
  args.padding = padding;

  p_driver->template pmap2d_read_coalesce<_fpmap_id,_fmap_lower>
    (output, input, args);

  report_last_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_lowering);
}

template<typename DataType, LayoutType InputLayout, typename DriverClass>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE1, DriverClass>::
remap_output(LogicalCube<DataType, InputLayout>& cube, const size_t R, const size_t C,
        const size_t kernel_size) {

  DeviceMemoryPointer * copy = p_driver->get_device_pointer(NULL, sizeof(DataType)*cube.R*cube.C*cube.B*cube.D);
  p_driver->malloc(copy);
  DeviceMemoryPointer * output = cube.get_device_pointer(p_driver);
  p_driver->memcpy(copy, output);

  static_assert(std::is_same<DataType, float>::value,
            "The func_src_to_dst function needs to change when DataType <> float.");

  PMapHelper args;
  args.dR = cube.R; args.dC = cube.C; args.dD = cube.D; args.dB = cube.B;
  args.sR = cube.R; args.sC = cube.C; args.sD = cube.D; args.sB = cube.B;
  args.dBR = args.dR; args.dBC = args.dC;
  args.sBR = min((size_t)32, args.sR); args.sBC = min((size_t)32, args.sC);
  args.stride = stride;
  args.padding = padding;

  p_driver->template pmap2d_read_coalesce<_fpmap_id,_fmap_remap>(output, copy, args);

  p_driver->free(copy);
  free(copy);
}

template<typename DataType, LayoutType InputLayout, typename DriverClass>
void Connector<DataType, InputLayout, DataType, Layout_CRDB, LOWERING_TYPE1, DriverClass>::
inverse_lower_cube(OutputLogicalCubeType * p_output_cube, InputLogicalCubeType * p_input_cube) {

  report_last_inverse_lowering.reset();

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

  DeviceMemoryPointer * input = p_input_cube->get_device_pointer(p_driver);
  // DeviceMemoryPointer * output = p_output_cube->get_device_pointer(p_driver);

  p_driver->sconstant_initialize(input, DataType(0.0));

  // const size_t data_output_width = (iR + 2 * padding - kernel_size) / stride + 1;  // the number of rows in the output gradient cube
  // const size_t data_output_height = (iC + 2 * padding - kernel_size) / stride + 1; // the number of cols in the output gradient cube

  assert(false);

  /*
  _func_src_to_dst_arg_helper_ilower1 arg1;

  arg1.iD = iD;
  arg1.iR = iR;
  arg1.iC = iC;
  arg1.iB = iB;
  arg1.kernel_size = kernel_size;
  arg1.padding = padding;
  arg1.stride = stride;
  arg1.oC = oC;
  arg1.data_output_width = data_output_width;
  arg1.data_output_height = data_output_height;

  DeviceMemoryPointer * parg1 = p_driver->get_device_pointer((void*)&arg1,
    sizeof(_func_src_to_dst_arg_helper_ilower1));

  DeviceMemoryPointer * parg2 = p_driver->get_device_pointer((void*)&arg1,
    sizeof(_func_src_to_dst_arg_helper_ilower1));

  p_driver->parallel_map(output, input, iR*iC*sizeof(DataType),
    &func_src_to_dst_conv_ilowering, parg1, &func_ilowering, parg2);
  */

  report_last_inverse_lowering.end(iR*iC*iD*iB*sizeof(DataType), oR*oC*oD*oB*sizeof(DataType), 0);
  report_history.aggregate(report_last_inverse_lowering);
}

#endif
