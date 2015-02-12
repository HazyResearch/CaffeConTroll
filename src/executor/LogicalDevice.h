//
//  LogicalDevice.h
//  moka
//
//  Created by Ce Zhang on 1/31/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LogicalDevice_h
#define moka_LogicalDevice_h

/**
A logical device is the only way a PhysicalDevice exposes
itself to the scheduler. A PhysicalDevice device contains a 
set of LogicalDevice, each of which can be executed in 
parallel.

Each logical device 

**/
class LogicalDevice{
public:

	const int DeviceType;
	const size_t memory_size;

	void * mem_transfer(PhysicalDevice * _device, void * const _memory, size_t size);

	void * 

};



#endif























