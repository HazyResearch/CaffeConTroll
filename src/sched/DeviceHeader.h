
#ifndef _DEVICE_HEADER_H
#define _DEVICE_HEADER_H


#include "DeviceMemoryPointer.h"
#include "cblas.h"


/**
 * We use C style function pointer here just becauses
 * there is no clean way in CUDA (7.0) and OpenCL (2.0)
 * to pass a host C++11 Lambda with capture to the device.
 * We could have used C-style macro to achieve this,
 * but that is even more messier.
 **/
typedef size_t (*FUNC_IDX_MAPPING) (size_t, void * const);
typedef void (*FUNC_MM_MAPPING) (void *, void *, void * const);
typedef float (*FUNC_STRANSFORM) (float, void * const);
typedef float (*FUNC_SREDUCE) (float, float, void * const);

struct PMapHelper{
  size_t dR, dC, dD, dB;  // dst RCDB
  size_t sR, sC, sD, sB;  // src RCDB
  size_t dBR, dBC;  // dst block
  size_t sBR, sBC;  // src block

  // lowering
  size_t kR, kC, kD, kB;  // kernel RCDB
  size_t stride;
  size_t padding;
};

struct Block2D{
  size_t r, c, d, b;
  size_t dr, dc;
} ;

struct PointIn2DBlock{
  float data;
  size_t r, c;
  Block2D block;
} ;

typedef void (*FPMAP_ID) (Block2D * const dst , const Block2D * const src, const PMapHelper * const args);
typedef void (*FPMAP_DATA_READC) (float * output, const Block2D * const output_block, const PointIn2DBlock * const input_point, 
  const PMapHelper * const args);



#endif