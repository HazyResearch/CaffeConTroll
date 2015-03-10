
#include "../sched/DeviceDriver_GPU.h"

#ifndef _LOWERING_INCLUDE_H
#define _LOWERING_INCLUDE_H

void invoke_lowering(GPUDriver * pdriver, DeviceMemoryPointer * dst, DeviceMemoryPointer * src, const struct PMapHelper args);

#endif