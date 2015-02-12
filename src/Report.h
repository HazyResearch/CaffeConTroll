//
//  Report.h
//  moka
//
//  Created by Ce Zhang on 1/12/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Report_h
#define moka_Report_h

#include "timer.h"

class Report {
public:

    size_t n_floating_point_ops;
    size_t n_data_read_byte;
    size_t n_data_write_byte;

    double elapsed_time;

    Timer t;

    Report() {
        reset();
    }

    double get_data_GB() {
        return 1.0*(n_data_read_byte+n_data_write_byte)/1024/1024/1024;
    }

    double get_throughput_GB() {
        return 1.0*(n_data_read_byte+n_data_write_byte)/elapsed_time/1024/1024/1024;
    }

    double get_flop_GFlop() {
        return 1.0*n_floating_point_ops/1024/1024/1024;
    }

    double get_flops_GFlops() {
        return 1.0*n_floating_point_ops/elapsed_time/1024/1024/1024;
    }

    void reset() {
        n_floating_point_ops = 0;
        n_data_read_byte = 0;
        n_data_write_byte = 0;
        elapsed_time = 0.0;
        t.restart();
    }

    void start() {
        t.restart();
    }

    void end(size_t _n_data_read_byte, size_t _n_data_write_byte, size_t _n_floating_point_ops) {
        elapsed_time += t.elapsed();
        n_floating_point_ops += _n_floating_point_ops;
        n_data_read_byte += _n_data_read_byte;
        n_data_write_byte += _n_data_write_byte;
    }

    void end() {
        elapsed_time += t.elapsed();
    }

    void aggregate_onlystat(const Report & report) {
        n_floating_point_ops += report.n_floating_point_ops;
        n_data_write_byte += report.n_data_write_byte;
        n_data_read_byte += report.n_data_read_byte;
    }

    void aggregate(const Report & report) {
        elapsed_time += report.elapsed_time;
        n_floating_point_ops += report.n_floating_point_ops;
        n_data_write_byte += report.n_data_write_byte;
        n_data_read_byte += report.n_data_read_byte;
    }

    void print() {
        //std::cout << "############REPORT#############" << std::endl;
        //std::cout << " Data Read+Write = " << get_data_GB() << " GBytes" << std::endl;
        //std::cout << " Float Point Ops = " << get_flop_GFlop() << " G" << std::endl;
        std::cout << " Time Elapsed    = " << elapsed_time << " seconds" << std::endl;
        std::cout << " Data Throughput = " << get_throughput_GB() << " GBytes/s" << std::endl;
        std::cout << " Flops           = " << get_flops_GFlops() << " GBytes/s" << std::endl;
        //std::cout << "###############################" << std::endl;
    }

};

#endif
