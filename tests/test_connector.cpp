#include "../src/Connector.h"
#include "../src/LogicalCube.h"
#include "test_types.h"
#include <cstring>
#include "gtest/gtest.h"
#include "glog/logging.h"
#include <iostream>
#include <fstream>
#include <assert.h>

using namespace std;

template<typename T, LayoutType LAYOUT>
void simple_lowering(LogicalCube<T, LAYOUT>* in, LogicalCube<T, Layout_CRDB>* out, int k, int s){
  int outc,outr=0;
  for(size_t kd=0;kd<in->D;kd++) {
    for(size_t kr=0;kr<k;kr++) {
      for(size_t kc=0;kc<k;kc++) {
        outc = 0;
        for(size_t ib=0;ib<in->B;ib++) {
          for(size_t cr=0;cr<(in->R-k)/s+1;cr++) {
            for(size_t cc=0;cc<(in->C-k)/s+1;cc++) {
              *out->logical_get(outr, outc, 0, 0) =
                *in->logical_get(cr*s+kr, cc*s+kc, kd, ib);
              outc ++;
            }
          }
        }
        outr ++;
      }
    }
  }
}

template <typename TypeParam>
class ConnectorLoweringType1Test : public ::testing::Test {
  protected:
    typedef typename TypeParam::T T;
    ConnectorLoweringType1Test(){
      input_cube = new LogicalCube<T, TypeParam::LAYOUT>(R,C,D,B);

      output_cube = new LogicalCube<T, Layout_CRDB>(k*k*D,((R-k)/s+1)*((C-k)/s+1)*B,1,1);
      connector_ = new Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, LOWERING_TYPE1>(input_cube, output_cube,
          k, p, s);
    }

    virtual ~ConnectorLoweringType1Test() { delete connector_; }
    Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, LOWERING_TYPE1>*  connector_;
    LogicalCube<T, TypeParam::LAYOUT>* input_cube;
    LogicalCube<T, Layout_CRDB>* output_cube;
    const int R = 4;
    const int C = 4;
    const int D = 1;
    const int B = 1;
    const int k = 2;
    const int s = 1;
    const int p = 0;
};


typedef ::testing::Types<FloatCRDB> DataTypes;

TYPED_TEST_CASE(ConnectorLoweringType1Test, DataTypes);

TYPED_TEST(ConnectorLoweringType1Test, TestInitialization) {
  EXPECT_TRUE(this->connector_);
}

TYPED_TEST(ConnectorLoweringType1Test, TestLowering){
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
    EXPECT_NEAR(this->output_cube->get_p_data()[i],  expected_output->get_p_data()[i], EPS);
  }

}

template <typename TypeParam>
class ConnectorLoweringType2Test : public ::testing::Test {
  protected:
    typedef typename TypeParam::T T;
    ConnectorLoweringType2Test(){
      input_cube = new LogicalCube<T, TypeParam::LAYOUT>(R,C,D,B);

      output_cube = new LogicalCube<T, Layout_CRDB>(B*R*C, D, 1, 1);
      connector_ = new Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, LOWERING_TYPE2>(input_cube, output_cube,
          k, p, s);
    }

    virtual ~ConnectorLoweringType2Test() { delete connector_; }
    Connector<T, TypeParam::LAYOUT, T, Layout_CRDB, LOWERING_TYPE2>*  connector_;
    LogicalCube<T, TypeParam::LAYOUT>* input_cube;
    LogicalCube<T, Layout_CRDB>* output_cube;
    const int R = 2;
    const int C = 2;
    const int D = 3;
    const int B = 4;
    // same as R and C
    const int k = 2;
    // not used in Lowering Type 2
    const int s = 0;
    const int p = 0;
};


TYPED_TEST_CASE(ConnectorLoweringType2Test, DataTypes);

TYPED_TEST(ConnectorLoweringType2Test, TestInitialization) {
  EXPECT_TRUE(this->connector_);
}

TYPED_TEST(ConnectorLoweringType2Test, TestLowering){
  typedef typename TypeParam::T T;
  int R = this->R;
  int C = this->C;
  int D = this->D;
  int B = this->B;
  int oR = B*R*C;
  int oC = D;

  fstream expected_input("tests/input/connector_lowering_2.txt", ios_base::in);
  T input; int idx = 0;
  if (expected_input.is_open()) {
    while (expected_input >> input) {
      this->input_cube->get_p_data()[idx++] = input;
    }
  } else {
    FAIL();
  }
  expected_input.close();
  EXPECT_NEAR(this->input_cube->n_elements, idx, 0.);

  LogicalCube<T, Layout_CRDB> * expected_output_cube = new LogicalCube<T, Layout_CRDB>(oR, oC, 1, 1);
  fstream expected_output("tests/output/connector_lowering_2.txt", ios_base::in);
  T output; idx = 0;
  if (expected_output.is_open()) {
    while (expected_output >> output) {
      expected_output_cube->get_p_data()[idx++] = output;
    }
  } else {
    FAIL();
  }
  expected_output.close();
  EXPECT_NEAR(expected_output_cube->n_elements, idx, 0.);

  int n = this->output_cube->n_elements;
  this->connector_->lower_model_cube(this->input_cube, this->output_cube);

  for(int i=0; i<n; i++){
    EXPECT_NEAR(this->output_cube->get_p_data()[i],  expected_output_cube->get_p_data()[i], EPS);
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

