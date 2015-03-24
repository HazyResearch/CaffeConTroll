
#ifndef _KERNEL_LRN_H
#define _KERNEL_LRN_H

#include "../sched/DeviceHeader.h"

struct _lrn_forward_arg_helper {
};

#ifdef _GPU_TARGET
__host__ __device__
#endif
size_t _f_src_to_dst_lrn_forward(size_t src_pos, void * const _arg);

#ifdef _GPU_TARGET
__host__ __device__
#endif
void _f_lrn_forward(void * bias, void * output, void * const _arg);

#endif
