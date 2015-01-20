#ifndef TEST_TYPES_H_
#define TEST_TYPES_H_

#include "../src/Cube.h"
#include "../src/Connector.h"
#include "../src/Scanner.h"
#include "../src/Kernel.h"

using std::cout;
using std::endl;

const double EPS=1e-4;

struct FloatCRDB{
	typedef DataType_SFFloat T;
	static const LayoutType LAYOUT = Layout_CRDB;
};

struct FloatBDRC{
	typedef DataType_SFFloat T;
	static const LayoutType LAYOUT = Layout_BDRC;
};

struct FloatNOFUNC{
	typedef DataType_SFFloat T;
	static const NonLinearFunction FUNC = FUNC_NOFUNC;
};

struct FloatTANH{
	typedef DataType_SFFloat T;
	static const NonLinearFunction FUNC = FUNC_TANH;
};

// struct FloatBDRC_FUNC_NOFUNC{
// 	typedef DataType_SFFloat T;
// 	static const LayoutType LAYOUT = Layout_CRDB;
// 	static const NonLinearFunction FUNC = FUNC_NOFUNC;
// };

// struct FloatCRDB_FUNC_NOFUNC{
// 	typedef DataType_SFFloat T;
// 	static const LayoutType LAYOUT = Layout_CRDB;
// 	static const NonLinearFunction FUNC = FUNC_NOFUNC;
// };

// struct FloatBDRC_FUNC_TANH{
// 	typedef DataType_SFFloat T;
// 	static const LayoutType LAYOUT = Layout_CRDB;
// 	static const NonLinearFunction FUNC = FUNC_TANH;
// };

// struct FloatCRDB_FUNC_TANH{
// 	typedef DataType_SFFloat T;
// 	static const LayoutType LAYOUT = Layout_CRDB;
// 	static const NonLinearFunction FUNC = FUNC_TANH;
// };


#endif