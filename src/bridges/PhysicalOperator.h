//
//  PhysicalOperator.h
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_PhysicalOperator_h
#define moka_PhysicalOperator_h
#include "../Report.h"

/**
 * A PhysicalOperator is a general
 * interface that has forward(), backward(),
 * and a set of common Report objects.
 **/
class PhysicalOperator {
public:
    int run_on_numa_node;
    int run_with_n_threads;

    Report report_forward_constructor;
    Report report_forward_last_transfer;
    Report report_forward_history;

    Report report_backward_updateweight_constructor;
    Report report_backward_updateweight_last_transfer;
    Report report_backward_updateweight_history;

    virtual void forward() = 0;
    virtual void backward() = 0;
};

#endif
