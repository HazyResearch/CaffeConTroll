#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>


using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}