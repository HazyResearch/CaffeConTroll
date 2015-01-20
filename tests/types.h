#ifndef TEST_TYPES_HPP_
#define TEST_TYPES_HPP_

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