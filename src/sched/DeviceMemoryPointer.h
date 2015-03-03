
#ifdef _WITH_CUDA
#include <cuda_runtime.h>
#endif

#ifndef _H_DeviceMemoryPointer_H
#define _H_DeviceMemoryPointer_H

enum DeviceMemoryType{
	DEVICEMEMORY_LOCAL_RAM = 0,
	DEVICEMEMORY_QPI_RAM   = 1,	// TODO
	DEVICEMEMORY_LOCAL_GPURAM = 2,
	DEVICEMEMORY_REMOTE_RAM = 3 // TODO
};

/**
 * A DeviceMemoryPointer is a pointer to 
 * a certain device, where device could
 * be local memory, cross-numa memory,
 * GPU memory, or remote memory, etc.
 *
 * Each subclass of DeviceMemoryPointer
 * needs to implement an interface
 * that allows memcpy to other devices, 
 * possibly in an DMA way. This memcpy, however,
 * is tranparent to the other device, in whose
 * eye this is modelled as pointer dereferencing.
 *
 * DeviceMemoryPointer is only a wrapper
 * of device-independent data access, so it
 * does not have the responsibility of freeing
 * memories that it points to. However, it has
 * the responsibility of freeing memory that it
 * allocates.
 *
 * Concurrent behavior and thread safety: For now,
 * it is the invoker's responsiblity to make sure
 * the access to this class is single-threaded.
 * This behavior might change after more implementation
 * is done.
 *
 * TODO:
 *   - interface of free'ed memory
 *   - interface of pinned memory for GPU performance 
 **/
class DeviceMemoryPointer{
public:

	DeviceMemoryPointer(void * _ptr, size_t _size_in_byte) :
		ptr(_ptr), size_in_byte(_size_in_byte){}

	/**
	 * Dereference the DeviceMemoryPointer to a memory
	 * pointer that the invoker knows how to manipulate.
	 * The resulting pointer could point to a local
	 * buffer of this class, or just a direct redirect.
	 *
	 * The pointer returned by p_device can be read
	 * and write by the invoker directly. However, the
	 * write will not guarantee'ed to be write through
	 * before `write_through` is called.
	 *
	 * If the deref function is called multiple times,
	 * the returned 'p_device' in different calles might
	 * point to same or different places. For now, I
	 * did not see any necesasarity in defining this behavior,
	 * but this decision might change during implementation.
	 *
	 **/
	virtual void deref_to(DeviceMemoryPointer * p_device) = 0;

	/**
	 * Synchronize the change to a deref'ed pointer
	 * to the device memory.
	 **/
	virtual void write_through(DeviceMemoryPointer * p_device) = 0;

protected:
	void * ptr;
	size_t size_in_byte;
	DeviceMemoryType type;
};

/**
 * A pointer that points to local RAM of the same NUMA node.
 * This is the most trivial case, and it is no more than
 * a thin wrapper over standard pointer.
 *
 **/
class DeviceMemoryPointer_Local_RAM : public DeviceMemoryPointer{
public:

	DeviceMemoryPointer_Local_RAM(void * _ptr, size_t _size_in_byte):
		type(DEVICEMEMORY_LOCAL_RAM), DeviceMemoryPointer(_ptr, _size_in_byte){}

	void deref_to(DeviceMemoryPointer * p_device){
		switch(p_device->type){
			case DEVICEMEMORY_LOCAL_RAM:
				// LOCAL RAM to LOCAL RAM is just copying pointers.
				p_device->ptr = this->_ptr;
				p_device->size_in_byte = this->size_in_byte;
			break;
			case DEVICEMEMORY_QPI_RAM: assert(false); // TODO
			case DEVICEMEMORY_LOCAL_GPURAM: 
#ifdef _WITH_CUDA
				// LOCAL RAM to LOCAL GPU MEMORY is DMA
				cudaError_t rs;

				// Allocate Device Memory
				rs = cudaMalloc((void**)&(p_device->ptr), this->size_in_byte);
				assert(rs == cudaSuccess);

				// DMA from Local Memory to Device
				cudaMemcpy(p_device->ptr, this->ptr, this->size_in_byte, cudaMemcpyHostToDevice);
				assert(rs == cudaSuccess);

				p_device->size_in_byte = this->size_in_byte;
#else
				std::cerr << "Error: Need _WITH_CUDA enabled!" << std::endl;
				assert(false);
#endif
			break
			case DEVICEMEMORY_REMOTE_RAM: assert(false); // TODO
		}
	}

	void write_through(DeviceMemoryPointer * p_device){
		switch(p_device->type){
			case DEVICEMEMORY_LOCAL_RAM: break;
			case DEVICEMEMORY_QPI_RAM: assert(false); // TODO
			case DEVICEMEMORY_LOCAL_GPURAM: 
#ifdef _WITH_CUDA
			cudaMemcpy(this->ptr, p_device->ptr, this->size_in_byte, cudaMemcpyHostToDevice);
			assert(rs == cudaSuccess);
#else
			std::cerr << "Error: Need _WITH_CUDA enabled!" << std::endl;
			assert(false);
#endif
			break
			case DEVICEMEMORY_REMOTE_RAM: assert(false); // TODO
		}
	}

private:
	using DeviceMemoryPointer::ptr;
	using DeviceMemoryPointer::size_in_byte;
	using DeviceMemoryPointer::type;
};

/**
 * A pointer that points to local GPU's RAM. 
 *
 * For now, I think we never deref a GPU memory TO 
 * other devices, or update GPU memory with worker 
 * on other devices. That is why the deref_to and
 * write_through function is empty here.
 * 
 **/
class DeviceMemoryPointer_Local_GPURAM : public DeviceMemoryPointer{
public:

	DeviceMemoryPointer_Local_GPURAM(int _GPUID, void * _ptr, size_t _size_in_byte):
		GPUID(_GPUID),
		type(DEVICEMEMORY_LOCAL_GPURAM), DeviceMemoryPointer(_ptr, _size_in_byte){
		assert(GPUID==0); // TODO: multiple GPUs
	}

	void deref_to(DeviceMemoryPointer * p_device){
		switch(p_device->type){
			case DEVICEMEMORY_LOCAL_RAM: assert(false); // TODO
			case DEVICEMEMORY_QPI_RAM: assert(false);  // TODO
			case DEVICEMEMORY_LOCAL_GPURAM: assert(false); // TODO
			case DEVICEMEMORY_REMOTE_RAM: assert(false); // TODO
		}
	}

	void write_through(DeviceMemoryPointer * p_device){
		switch(p_device->type){
			case DEVICEMEMORY_LOCAL_RAM: assert(false); // TODO
			case DEVICEMEMORY_QPI_RAM: assert(false); // TODO
			case DEVICEMEMORY_LOCAL_GPURAM: assert(false); // TODO
			case DEVICEMEMORY_REMOTE_RAM: assert(false); // TODO
		}
	}

private:
	using DeviceMemoryPointer::ptr;
	using DeviceMemoryPointer::size_in_byte;
	using DeviceMemoryPointer::type;

	int GPUID;
};


#endif






