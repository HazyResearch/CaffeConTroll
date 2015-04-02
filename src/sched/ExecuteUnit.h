
#include "DeviceMemoryPointer.h"

#ifndef _EXECUTE_UNIT_H
#define _EXECUTE_UNIT_H

/**
 * An ExecuteUnit is a abstract object
 * that models all computation and data
 * manipulation in CcT that follows the 
 * following pattern.
 *   - Input: A set of DeviceMemoryPointer.
 *   - Output: write_through (see DeviceMemoryPointer) changes
 *     to a subset of DeviceMemoryPointer.
 *   - Operation:
 *      1. Deref Input to memory that ExecuteUnit can manipulate.
 *         This might be a single memory redirect, or more expensive
 *         data movement.
 *      2. Execute.
 *      3. Write Through to input memories.
 * Each operation corresponds to one function, namely
 * load(), execute(), finish(). All subclass of ExecuteUnit
 * needs to expect sequence of calles in the order of
 * blocked calls load() -> execute() -> finish(). In other word,
 * an ExecuteUnit does not need to worry about out-of-order or
 * asynchronized calls of these three functions. 
 *
 * Note that, an ExecuteUnit might not be the smallest ExecuteUnit,
 * that is, an ExecuteUnit might be a sequence of smaller ExecuteUnits,
 * or a set of sequences of ExecuteUnits that runs in parallel with
 * double buffering. However, the load->execute->finish sequence does
 * not change.
 *
 * All higher-level objects, e.g., ConvolutionBridge, Kernel, Connector,
 * have only a single responsibility, that is emits a proper ExecuteUnit.
 * The Whole CcT run itself is a single ExecuteUnit whose input are images, 
 * and output are master-copy of the models.
 **/
class ExecuteUnit{
public:
	virtual load() = 0;
	virtual execute() = 0;
	virtual finish() = 0;
};

#endif