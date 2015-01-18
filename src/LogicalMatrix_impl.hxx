//
//  LogicalMatrix_impl.hxx
//  moka
//
//  Created by Firas Abuzaid on 1/16/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_LogicalMatrix_impl_hxx
#define moka_LogicalMatrix_impl_hxx

/** Constructor (takes in extra argument to point to another data block) **/
template <typename T>
LogicalMatrix<T>::LogicalMatrix(T * _p_data, size_t _R, size_t _C) :
p_data(_p_data),
n_elements(_R*_C),
R(_R), C(_C)
{}

/** Destructor **/
template <typename T>
LogicalMatrix<T>::~LogicalMatrix(){}

template <typename T>
void LogicalMatrix<T>::physical_print() const {
    for(size_t i = 0; i < n_elements; ++i) {
      std::cout << p_data[i] << " ";
    }
    std::cout << std::endl;
}

#endif


