#include "PhysicalStratum.h"

/**
 * Wrapper of calling the forward() function -- only
 * used when we call thread
 **/
void _forward(PhysicalOperator * const bridge) {
  bridge->forward();
}

void _backward(PhysicalOperator * const bridge) {
  bridge->backward();
}

PhysicalStratum::PhysicalStratum() {

  report_forward_constructor.reset();
  report_forward_last_transfer.reset();
  report_forward_history.reset();
  report_forward_constructor.end(0, 0, 0);

  report_backward_updateweight_constructor.reset();
  report_backward_updateweight_last_transfer.reset();
  report_backward_updateweight_history.reset();
  report_backward_updateweight_constructor.end(0, 0, 0);
}

void PhysicalStratum::forward() {
  report_forward_last_transfer.reset();
  // We could build a thread pool; however, we are talking about
  // at most 20 threads / 10 seconds etc.
  // TODO: benchmark to see whether we want more sophisticated
  //       thread pool.
  vector<thread> threads;
  for (size_t i = 0; i < executor_bound; i++) {
    threads.push_back(thread(_forward, executors[i]));
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

void PhysicalStratum::backward() {
  report_backward_updateweight_last_transfer.reset();
  vector<thread> threads;

  for (size_t i = 0; i < executor_bound; i++) {
    threads.push_back(thread(_backward, executors[i]));
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
