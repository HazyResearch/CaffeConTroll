#ifndef TEST_MAIN_HPP_
#define TEST_MAIN_HPP_


// #include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "test_main.h"

using std::cout;
using std::endl;

struct FloatCRDB{
	typedef DataType_SFFloat T;
	static const LayoutType LAYOUT = Layout_CRDB;
};

struct FloatBDRC{
	typedef DataType_SFFloat T;
	static const LayoutType LAYOUT = Layout_BDRC;
};

#endif