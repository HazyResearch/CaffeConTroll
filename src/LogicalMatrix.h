//
//  LogicalMatrix.h
//  moka
//
//  Created by Firas Abuzaid on 1/16/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//
//

#ifndef moka_LogicalMatrix_h
#define moka_LogicalMatrix_h

#include <assert.h>
#include <string>
#include <iostream>
#include "util.h"

//enum LogicalMatrixLayout {
//    ROW_MAJOR = 0,
//    COLUMN_MAJOR = 1
//};

/*
 * A single RxC slice of the LogicalCube. A LogicalMatrix m is retrieved from the given
 * LogicalCube by giving a batch index and a depth index
 * (Note: the LogicalMatrix class doesn't manage any memory; it has a pointer to
 * the underlying data in the LogicalCube)
 */
template <typename T>
class LogicalMatrix {
public:
    T * const p_data;
    const size_t n_elements;
    const size_t R;
    const size_t C;

    LogicalMatrix(T * _p_data, size_t _R, size_t _C);
    ~LogicalMatrix();

    void physical_print() const;
private:


};

#include "LogicalMatrix_impl.hxx"

#endif
