//
//  PhysicalStratum.h
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_PhysicalStratum_h
#define moka_PhysicalStratum_h

#include "AbstractBridge.h"
#include "PhysicalOperator.h"
#include <thread>
#include <vector>

using std::thread;
using std::vector;

/**
 * Wrapper of calling the forward() function -- only
 * used when we call thread
 **/
void _forward(PhysicalOperator * const bridge);

void _backward(PhysicalOperator * const bridge);


/**
 * A Stratum is a set of PhysicalOperators that will
 * be run in parallel. A Stratum itself is also
 * an PhysicalOperator.
 **/
class PhysicalStratum : public PhysicalOperator {
  protected:
    size_t executor_bound;

  public:
    vector<PhysicalOperator *> executors; // STL overhead is not that crucial here,
                                          // so we just use a vector
    PhysicalStratum();

    void forward();
    void backward();

    void set_executor_bound(size_t _executor_bound) {
      executor_bound = _executor_bound;
    }
};

#include "PhysicalStratum_impl.hxx"

#endif
