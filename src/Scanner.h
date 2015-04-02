//
//  Scanner.h
//  moka
//
//  Created by Ce Zhang on 1/14/15.
//  Copyright (c) 2015 Hazy Research. All rights reserved.
//

#ifndef moka_Scanner_h
#define moka_Scanner_h

#include "sched/DeviceDriver.h"

#include "LogicalCube.h"
#include "Report.h"

enum NonLinearFunction {
    FUNC_NOFUNC = 0,
    FUNC_TANH = 1
};

/**
 * A scanner is simple -- for each element in a LogicalCube, apply a function, and update the element.
 **/
template
<typename DataType, LayoutType Layout, NonLinearFunction SCANNER>
class Scanner {
public:

    typedef LogicalCube<DataType, Layout> LogicalCubeType;

    Report report_constructor; /*< Performance reporter for constructor function. */
    Report report_last_apply; /*< Performance reporter for the last run of transfer() function. */
    Report report_history; /*< Performance reporter for all transfer() functions aggregated. */

    DeviceDriver * p_driver;

    Scanner(const LogicalCubeType * const p_cube, DeviceDriver * _p_driver){
        std::cerr << "ERROR: Using a scanner with unsupported Layout or DataType." << std::endl;
        assert(false);
    }

    void apply(LogicalCubeType * const p_cube){
        std::cerr << "ERROR: Using a scanner with unsupported Layout or DataType." << std::endl;
        assert(false);
    }

};

/******
 * Specializations
 */
template
<typename DataType, LayoutType Layout>
class Scanner<DataType, Layout, FUNC_TANH>{
public:

    typedef LogicalCube<DataType, Layout> LogicalCubeType;

    Report report_constructor; /*< Performance reporter for constructor function. */
    Report report_last_apply; /*< Performance reporter for the last run of transfer() function. */
    Report report_history; /*< Performance reporter for all transfer() functions aggregated. */

    DeviceDriver * p_driver;

    Scanner(const LogicalCubeType * const p_cube, DeviceDriver * _p_driver);

    void apply(LogicalCubeType * const p_cube);

};

template
<typename DataType, LayoutType Layout>
class Scanner<DataType, Layout, FUNC_NOFUNC>{
public:

    typedef LogicalCube<DataType, Layout> LogicalCubeType;

    Report report_constructor; /*< Performance reporter for constructor function. */
    Report report_last_apply; /*< Performance reporter for the last run of transfer() function. */
    Report report_history; /*< Performance reporter for all transfer() functions aggregated. */

    DeviceDriver * p_driver;

    Scanner(const LogicalCubeType * const p_cube, DeviceDriver * _p_driver){
        report_constructor.reset();
        report_last_apply.reset();
        report_history.reset();

        report_constructor.end(0, 0, 0);
    }

    void apply(LogicalCubeType * const p_cube){}

};

#include "Scanner_impl.hxx"

#endif
