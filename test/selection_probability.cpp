#include "gtest/gtest.h"

#include "kernels.h"
#include "../include/kernels.h"
#include <Eigen/Dense>

#include <algorithm>
#include <memory>

using namespace dpp;

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


  auto ckern = wrap_ptr(c_kernel<double>::from_colwize_array(data, dim, row_cnt));
  auto sampler = wrap_ptr(ckern->sampler(3));

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

  auto lkern = wrap_ptr(l_kernel<double>::from_array(l_data.data(), row_cnt));

  auto selection = sampler->sample();

  std::vector<i64> sel2;
  auto rw = wrap_result(sel2);

  auto l_sampler = lkern->selector();
  l_sampler->greedy_max_subset(4, rw);

  std::cout << "[" << selection[0] << ", "
            << selection[1] << ", "
            << selection[2] << "]" << std::endl;

  std::cout << "[" << sel2[0] << ", "
  << sel2[1] << ", "
  << sel2[2] << ", " << sel2[3] << "]" << std::endl;

  auto c_prob = ckern->selection_log_probability(selection);
  auto l_prob = lkern->selection_log_probability(selection);
  EXPECT_FLOAT_EQ(c_prob, l_prob);
}

TEST_F(SelectionProbabilityTest, SameSelection) {
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

  Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(data, 11, 6);

  Eigen::MatrixXd m2 = m * m.adjoint();

  EXPECT_EQ(m2.rows(), 11);
  EXPECT_EQ(m2.cols(), 11);

  auto lkern = wrap_ptr(dpp::l_kernel<double>::from_array(m2.data(), 11));
  auto ckern = wrap_ptr(dpp::c_kernel<double>::from_colwize_array(m.data(), 6, 11));

  auto s1 = lkern->selector();
  auto s2 = ckern->selector();

  std::vector<i64> v1;
  std::vector<i64> v2;
  auto wr1 = wrap_result(v1);
  auto wr2 = wrap_result(v2);

  s1->greedy_max_subset(4, wr1);
  s2->greedy_max_subset(4, wr2);

  EXPECT_EQ(v1[0], v2[0]);
  EXPECT_EQ(v1[1], v2[1]);
  EXPECT_EQ(v1[2], v2[2]);
  EXPECT_EQ(v1[3], v2[3]);
}