//
// Created by Arseny Tolmachev on 2015/07/31.
//

#include "gtest/gtest.h"
#include "kernels.h"

class SelectorsTest: public ::testing::Test {
  const i64 dim = 6;
  std::vector<double> data_;
  i64 row_cnt;

public:
  SelectorsTest():
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
