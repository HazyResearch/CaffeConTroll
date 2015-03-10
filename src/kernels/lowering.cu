
#include "lowering_include.h"

void invoke_lowering(GPUDriver * pdriver, DeviceMemoryPointer * dst, DeviceMemoryPointer * src, const struct PMapHelper args){
	pdriver->pmap2d_read_coalesce<_fpmap_id,_fmap_lower>(dst, src, args);
}
