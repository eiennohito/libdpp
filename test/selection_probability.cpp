#include "gtest/gtest.h"

#include "kernels.h"
#include "../include/kernels.h"
#include <Eigen/Dense>

#include <algorithm>
#include <memory>

template <typename T>
std::unique_ptr<T> wrap_ptr(T *ptr) {
  return std::unique_ptr<T>(ptr);
}

class SelectionProbabilityTest: public ::testing::Test {};

TEST_F(SelectionProbabilityTest, ProbEqual) {
  const i64 dim = 6;
  double data[] =      {1, 0, 0, 0, 0.1, 0,
                        0, 1, 1, 0, 0, 0,
                        0, 0, 1, 0, 0, 0.2,
                        1, 1, 0, 0.1, 0, 0,
                        1, 1, 1, 0, 0, 0.1,
                        1, 0, 0.1, 0, 1, 0,
                        0.1, 1, 1, 0, 0.1, 0,
                        0.1, 1, 0.1, 0, 0, 0,
                        1, 1, 1, 1, 1, 1,
                        1, 0, 0, 0, 1, 1,
                        0, 0.1, 0, 1, 1, 1};

    i64 row_cnt = sizeof(data) / (sizeof(double) * dim);


  auto c_kernel = wrap_ptr(dpp::c_kernel<double>::from_colwize_array(data, dim, row_cnt));
  auto sampler = wrap_ptr(c_kernel->sampler(3));

  std::vector<double> l_data;

  for (i64 i = 0; i < row_cnt; ++i) {
    for (i64 j = 0; j < row_cnt; ++j) {
      double v = 0;
      for (i64 k = 0; k < dim; ++k) {
        v += data[i * dim + k] * data[j * dim + k];
      }
      l_data.push_back(v);
    }
  }

  auto l_kernel = dpp::l_kernel<double>::from_array(l_data.data(), row_cnt);


  auto selection = sampler->sample();

  std::cout << "[" << selection[0] << ", "
            << selection[1] << ", "
            << selection[2] << "]" << std::endl;

  auto c_prob = c_kernel->selection_log_probability(selection);
  auto l_prob = l_kernel->selection_log_probability(selection);
  EXPECT_FLOAT_EQ(c_prob, l_prob);

}