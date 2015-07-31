//
// Created by Arseny Tolmachev on 2015/07/31.
//

#include "gtest/gtest.h"
#include "libdpp_c.h"

class CApiTest: public ::testing::Test {
protected:
  const i64 dim = 6;
  std::vector<double> data_;
  i64 row_cnt;
public:
  CApiTest():
      data_{
          1, 0, 0, 0, 0.1, 0,
          0, 1, 1, 0, 0, 0,
          0, 0, 1, 0, 0, 0.2,
          1, 1, 0, 0.1, 0, 0,
          1, 1, 1, 0, 0, 0.1,
          1, 0, 0.1, 0, 1, 0,
          0.1, 1, 1, 0, 0.1, 0,
          0.1, 1, 0.1, 0, 0, 0,
          1, 1, 1, 1, 1, 1,
          1, 0, 0, 0, 1, 1,
          0, 0.1, 0, 1, 1, 1
      }
  {
    row_cnt = (i64) (data_.size() / dim);
  }
};


TEST_F(CApiTest, SimpleWorkflowWorks) {
  dpp_c_kernel kernel = dpp_create_c_kernel(data_.data(), dim, row_cnt);

  i64 out[4];
  dpp_select_from_c(kernel, 4, out);
  EXPECT_EQ(10, out[0]);

  dpp_delete_c_kernel(kernel);
}