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
    PhysicalStratum() {
      report_forward_constructor.reset();
      report_forward_last_transfer.reset();
      report_forward_history.reset();
      report_forward_constructor.end(0, 0, 0);

      report_backward_updateweight_constructor.reset();
      report_backward_updateweight_last_transfer.reset();
      report_backward_updateweight_history.reset();
      report_backward_updateweight_constructor.end(0, 0, 0);
    }

    void forward() {
      report_forward_last_transfer.reset();
      // We could build a thread pool; however, we are talking about
      // at most 20 threads / 10 seconds etc.
      // TODO: benchmark to see whether we want more sophisticated
      //       thread pool.

      // SHADJIS TODO: If there is just 1 thread we don't need
      // to launch a new thread here?
      
      vector<thread> threads;
      for (size_t i = 0; i < executor_bound; i++) {
        //threads.push_back(thread(_forward, executors[i]));
        threads.push_back(thread([this, i]()
              {
              executors[i]->forward();
              }));
      }
      for (size_t i = 0; i < executor_bound; i++) {
        threads[i].join();
      }
      report_forward_last_transfer.end();

      for (size_t i = 0; i < executor_bound; i++) {
        report_forward_last_transfer.aggregate_onlystat(executors[i]->report_forward_last_transfer);
      }
      report_forward_history.aggregate(report_forward_last_transfer);
    }

    void backward() {
      report_backward_updateweight_last_transfer.reset();
      vector<thread> threads;

      for (size_t i = 0; i < executor_bound; i++) {
        // threads.push_back(thread(_backward, executors[i]));
        threads.push_back(thread([this, i]()
              {
              executors[i]->backward();
              }));
      }
      for (size_t i = 0; i < executor_bound; i++) {
        threads[i].join();
      }
      report_backward_updateweight_last_transfer.end();

      for (size_t i = 0; i < executor_bound; i++) {
        report_backward_updateweight_last_transfer.aggregate_onlystat(executors[i]->report_backward_updateweight_last_transfer);
      }
      report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
    }

    void set_executor_bound(size_t _executor_bound) {
      executor_bound = _executor_bound;
    }
};

// #include "PhysicalStratum_impl.hxx"

#endif
