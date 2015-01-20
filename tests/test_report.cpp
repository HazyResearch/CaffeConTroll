#include <assert.h>
#include <cmath>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "test_types.h"
#include "../src/Report.h"

using namespace std;

#define show(x) std::cerr << #x << " : " << x << endl;

class ReportTest : public ::testing::Test{
public:
	ReportTest(){
		r_.n_floating_point_ops=1e7;
		r_.n_data_read_byte=1e6;
		r_.n_data_write_byte=1e8;
		r_.elapsed_time=2.0;
	}

	double bytes_to_GBytes(float b){
		return b/1024/1024/1024;
	}
	Report r_;
};

TEST_F(ReportTest, TestReportGetDataGB){
	ASSERT_NEAR(r_.get_data_GB(), bytes_to_GBytes(r_.n_data_read_byte+r_.n_data_write_byte), EPS);	
}

TEST_F(ReportTest, TestReportGetThroughputGB){
	ASSERT_NEAR(r_.get_throughput_GB(), bytes_to_GBytes(r_.n_data_read_byte+r_.n_data_write_byte)/r_.elapsed_time , EPS);
}

TEST_F(ReportTest, TestReportGetFlopGFlop){
	ASSERT_NEAR(r_.get_flop_GFlop(), bytes_to_GBytes(r_.n_floating_point_ops), EPS);
}

TEST_F(ReportTest, TestReportGetFlopsGFlops){
	ASSERT_NEAR(r_.get_flops_GFlops(), bytes_to_GBytes(r_.n_floating_point_ops)/r_.elapsed_time, EPS);
}

TEST_F(ReportTest, TestReportReset){
	Report r=r_;

	r.reset();

	ASSERT_NEAR(r.t.elapsed(), 0.0, EPS);
	EXPECT_EQ(r.n_floating_point_ops, 0);
	EXPECT_EQ(r.n_data_read_byte, 0);
	EXPECT_EQ(r.n_data_write_byte, 0);
	EXPECT_EQ(r.elapsed_time, 0.0);
}

TEST_F(ReportTest, TestReportStart){
	Report r=r_;

	r.t.restart();

	ASSERT_NEAR(r.t.elapsed(), 0.0, EPS);
}

TEST_F(ReportTest, TestReportEnd){
	Report r=r_;

	r.reset();

	r.n_floating_point_ops=1e7;
	r.n_data_read_byte=1e6;
	r.n_data_write_byte=1e8;
	r.elapsed_time=2.0;

	r.end(1e6, 1e8, 1e7);

	EXPECT_EQ(r.n_floating_point_ops, 2e7);
	EXPECT_EQ(r.n_data_read_byte, 2e6);
	EXPECT_EQ(r.n_data_write_byte, 2e8);
	ASSERT_NEAR(r.elapsed_time,2.0,EPS);
}

TEST_F(ReportTest, TestReportAggregate){
	Report r2;

	r2.n_floating_point_ops=1e7;
	r2.n_data_read_byte=2e6;
	r2.n_data_write_byte=3e8;
	r2.elapsed_time=4.0;

	r2.aggregate(r_);

	EXPECT_EQ(r2.n_floating_point_ops, 2e7);
	EXPECT_EQ(r2.n_data_read_byte, 3e6);
	EXPECT_EQ(r2.n_data_write_byte, 4e8);
	ASSERT_NEAR(r2.elapsed_time,6.0,EPS);
}






