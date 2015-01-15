//
//  PhysicalStratum_impl.hxx
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_PhysicalStratum_impl_hxx
#define moka_PhysicalStratum_impl_hxx

PhysicalStratum::PhysicalStratum(){
    report_forward_constructor.reset();
    report_forward_last_transfer.reset();
    report_forward_history.reset();
    report_forward_constructor.end(0, 0, 0);
        
    report_backward_updateweight_constructor.reset();
    report_backward_updateweight_last_transfer.reset();
    report_backward_updateweight_history.reset();
    report_backward_updateweight_constructor.end(0, 0, 0);
}
    
void PhysicalStratum::forward(){
    report_forward_last_transfer.reset();
    std::vector<std::thread> threads; // We could build a thread pool, however,
                                      // we are talking about at most 20 threads / 10 seconds etc.
                                      // TODO: benchmark to see whether we want more sophisticated
                                      //       thread pool.
    for(int i=0;i<executors.size();i++){
        threads.push_back(std::thread(_forward, executors[i]));
    }
    for(int i=0;i<executors.size();i++){
        threads[i].join();
    }
    report_forward_last_transfer.end();
    for(int i=0;i<executors.size();i++){
        report_forward_last_transfer.aggregate_onlystat(executors[i]->report_forward_last_transfer);
    }
    report_forward_history.aggregate(report_forward_last_transfer);
}
    
void PhysicalStratum::backward(){
    report_backward_updateweight_last_transfer.reset();
    std::vector<std::thread> threads;
    for(int i=0;i<executors.size();i++){
        threads.push_back(std::thread(_backward, executors[i]));
    }
    for(int i=0;i<executors.size();i++){
        threads[i].join();
    }
    report_backward_updateweight_last_transfer.end();
    for(int i=0;i<executors.size();i++){
        report_backward_updateweight_last_transfer.aggregate_onlystat(executors[i]->report_backward_updateweight_last_transfer);
    }
    report_backward_updateweight_history.aggregate(report_backward_updateweight_last_transfer);
}

#endif
