//
//  PhysicalDevice.h
//  moka
//
//  Created by Ce Zhang on 1/31/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_PhysicalDevice_h
#define moka_PhysicalDevice_h

enum PhysicalDeviceType{
	DEVICE_LOCAL_CPU = 0,
	DEVICE_LOCAL_GPU = 1,
	DEVICE_REMOTE_CPU = 2,
	DEVICE_REMOTE_GPU = 3
};

/**
A physical device contains two components, that is (1) a piece
of _on-device_ memory and (2) a set of computation cores. However,
not all (memory, cores) pair are eligible to be be a PhysicalDevice,
it must satisfy a set of requirements, which we will elaborate as 
follows.

In short, a PhysicalDevice is the _maximal_ (memory, cores) pair
that are collocated via a fast bus. As an example, following
things are PhysicalDevice:
  - A single local CPU, no matter how many cores it has, and the correspoinding
  memory that do not need to be access across QPI.
  - A single local GPU, with all its cores and all its device memory.
  - A single remote CPU.
  - A singel remote GPU.
Following things are NOT PhysicalDevice:
  - A NUMA machine with more than one NUMA node. 
  - An array of GPU.

Note that, a PhysicalDevice does not directly accept jobs. It exposes
itself to the scheduler with a set of views called LogicalDevice, which
abstracts and hides the difference between different PhysicalDevice to the
scheduler.

A PhysicalDevice provides to LogicalDevice its interface to access memory
from other PhysicalDevice, either though:
   - Local memory <-> local memory: pointer dereference;
   - Local memory <-> QPI <-> local memory: pointer dereference or copy through QPI.
   - Local memory <-> local GPU memory: DMA via PCIe bus
   - local GPU memory <-> local GPU memory: ____
   - Local memory <-> remote memory: DMA via network.
   - Local memory <-> remove GPU memory: ____

**/
template<PhysicalDeviceType>
class PhysicalDevice{
public:

	const int DeviceType;
	const size_t memory_size;

	void * mem_transfer(PhysicalDevice * _device, void * const _memory, size_t size);

	void * 

};



#endif























