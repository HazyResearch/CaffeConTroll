#ifndef TEST_MAIN_HPP_
#define TEST_MAIN_HPP_

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "test_main.h"
#include "../snapshot-parser/simple_parse.h"

using std::cout;
using std::endl;

struct FloatCRDB {
	typedef DataType_SFFloat T;
	static const LayoutType LAYOUT = Layout_CRDB;
};

struct FloatBDRC {
	typedef DataType_SFFloat T;
	static const LayoutType LAYOUT = Layout_BDRC;
};

void compare_to_expected(const LogicalCube<float, Layout_CRDB> * const actual, std::ostream & expected) {
    float output;
    int idx = 0;
    if (expected.is_open()) {
        while(expected >> output) {
            EXPECT_NEAR(actual->p_data[idx++], output, EPS);
        }
    } else {
        FAIL();
    }
}

void read_from_file(LogicalCube<float, Layout_CRDB> * cube, std::istream & input) {
    float element;
    int idx = 0;
    if (input.is_open()) {
        while (input >> element) {
            cube->p_data[idx++] = element;
        }
    } else {
        FAIL();
    }
}

#endif