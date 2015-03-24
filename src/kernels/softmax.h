
#ifndef _KERNEL_SOFTMAX_H
#define _KERNEL_SOFTMAX_H

#include "../sched/DeviceHeader.h"

struct _softmax_forward_arg_helper {
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_softmax_forward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_softmax_forward(void * bias, void * output, void * const _arg);

#endif
