//
//  PhysicalStratum.h
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_PhysicalStratum_h
#define moka_PhysicalStratum_h

#include "PhysicalOperator.h"
#include <thread>
#include <vector>

/**
 * Wrapper of calling the forward() function -- only
 * used when we call std::thread
 **/
void _forward(PhysicalOperator * physical_executor){
  physical_executor->forward();
}

void _backward(PhysicalOperator * physical_executor){
  physical_executor->backward();
}

/**
 * A Stratum is a set of PhysicalOperators that will
 * be run in parallel. A Stratum itself is also
 * an PhysicalOperator.
 **/
class PhysicalStratum : public PhysicalOperator{
  public:
    std::vector<PhysicalOperator*> executors; // STL overhead is not that crucial here, so
                                              // we just use a vector

    PhysicalStratum();

    void forward();

    void backward();
};

#include "PhysicalStratum_impl.hxx"

#endif
