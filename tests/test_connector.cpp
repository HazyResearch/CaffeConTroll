
#include "test_types.h"
#include <cstring>
#include "gtest/gtest.h"
#include "glog/logging.h"
#include <iostream>
#include <assert.h>

#include "../src/sched/DeviceDriver_CPU.h"
#include "../src/Connector.h"
#include "../src/LogicalCube.h"

using namespace std;

template<typename T, LayoutType LAYOUT>
// Simple implementation of Type 1 lowering
// Lowers entire rows at a time as described in the paper
void simple_lowering(LogicalCube<T, LAYOUT>* in, LogicalCube<T, Layout_CRDB>* out, int k, int s){
  
  assert(in->C == in->R);
  const int p = 0;
  const int m = (in->C + 2*p - k) / s + 1;
  
  for (int bi=0; bi<in->B; ++bi) {
    for (int r=0; r<m; ++r) {
      for (int c=0; c<m; ++c) {
        float *current_row = &(out->get_p_data()[(bi*m*m + r*m + c)*k*k*in->D]);
        for (int Dd=0; Dd<in->D; ++Dd) {
          for (int Dr=0; Dr<k; ++Dr) {
            for (int Dc=0; Dc<k; ++Dc) {
              if ( (r*s-p+Dr) >= 0 && (r*s-p+Dr) < in->R && (c*s-p+Dc) >= 0 && (c*s-p+Dc) < in->C ) {
                current_row[Dd*k*k + Dr*k + Dc] = in->get_p_data()[bi*in->R*in->C*in->D + Dd*in->R*in->C + (r*s-p+Dr)*in->C + (c*s-p+Dc)];
              } else {
                current_row[Dd*k*k + Dr*k + Dc] = 0;
              }
            }
          }
        }
      }
    }
  }
}

template <typename TypeParam>
class ConnectorTest : public ::testing::Test {
  protected:
    typedef typename TypeParam::T T;
    ConnectorTest(){
      input_cube = new LogicalCube<T, TypeParam::LAYOUT>(R,C,D,B);

      output_cube = new LogicalCube<T, Layout_CRDB>(k*k*D,((R-k)/s+1)*((C-k)/s+1)*B,1,1);
      connector_ = new Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, LOWERING_TYPE1, CPUDriver>(input_cube, output_cube,
          k, p, s, &pdriver);
    }

    virtual ~ConnectorTest() { delete connector_; }
    Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, LOWERING_TYPE1, CPUDriver>*  connector_;
    LogicalCube<T, TypeParam::LAYOUT>* input_cube;
    LogicalCube<T, Layout_CRDB>* output_cube;
    CPUDriver pdriver;
    const int R = 4;
    const int C = 4;
    const int D = 1;
    const int B = 1;
    const int k = 2;
    const int s = 1;
    const int p = 0;
};


typedef ::testing::Types<FloatCRDB> DataTypes;

TYPED_TEST_CASE(ConnectorTest, DataTypes);

TYPED_TEST(ConnectorTest, TestInitialization) {
  EXPECT_TRUE(this->connector_);
}

TYPED_TEST(ConnectorTest, TestLowering){
  typedef typename TypeParam::T T;
  int R = this->R;
  int C = this->C;
  int D = this->D;
  int B = this->B;
  int k = this->k;
  int s = this->s;
  int oR = (R-k)/s+1;
  int oC = (C-k)/s+1;

  for(int r=0;r<this->input_cube->R;r++){
    for(int c=0;c<this->input_cube->C;c++){
      for(int d=0;d<this->input_cube->D;d++){
        for(int b=0;b<this->input_cube->B;b++){
          *this->input_cube->logical_get(r,c,d,b) = rand()%10;
        }
      }
    }
  }

  LogicalCube<T, Layout_CRDB>* expected_output = new LogicalCube<T, Layout_CRDB>(k*k*D,oR*oC*B,1,1);
  simple_lowering<T,TypeParam::LAYOUT>(this->input_cube,expected_output,k,s);

  int n = this->output_cube->n_elements;
  this->connector_->lower_cube(this->input_cube, this->output_cube);

  for(int i=0; i<n; i++){
    EXPECT_NEAR(this->output_cube->get_p_data()[i],  expected_output->get_p_data()[i],EPS);
  }

}

//TODO -- Check for inverse lowering
/*
   TYPED_TEST(ConnectorTest, TestInverseLowering){
   for(int i=0; i<this->output_cube->n_elements; i++){
   this->output_cube->get_p_data()[i] = i;
   }


   typedef typename TypeParam::T T;
   LogicalCube<T, Layout_CRDB> input_expected(5, 5, 3, 1);

   this->connector_->inverse_lower_cube(this->output_cube, this->input_cube);

   cout << "Cube to be lowered" << endl;
   this->output_cube->logical_print();

   cout << "Inverse Lowered Cube" << endl;
   this->input_cube->logical_print();
   if (TypeParam::LAYOUT == Layout_CRDB){
   for(int i=0; i<this->output_cube->n_elements; i++){
 *input_expected.logical_get((i/24)%3+(i/2)%2, (i/8)%3+(i%2), i/72, (i/4)%2) += *this->output_cube->logical_get(i/8,i%8,0,0);
 }

 for(int i=0; i<this->input_cube->n_elements; i++){
//EXPECT_NEAR(input_expected.p_data[i], this->input_cube->get_p_data()[i],EPS);
}

}
}
*/

