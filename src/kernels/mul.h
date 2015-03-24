
#ifndef _KERNEL_MUL_H
#define _KERNEL_MUL_H

#include "../sched/DeviceHeader.h"

#ifdef _GPU_TARGET
__host__ __device__
#endif
float _f_reduce_mul(float a, float b, void * const);

#ifdef _GPU_TARGET
__host__ __device__
#endif
float _f_reduce_tanhgrad(float a, float b, void * const);

#endif