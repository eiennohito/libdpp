//
// Created by Arseny Tolmachev on 2015/07/31.
//

#include "gtest/gtest.h"
#include "kernels.h"

#include "../src/common.hpp"
#include "../src/l_kernel.hpp"
#include "../src/l_selection_impl.hpp"
#include "../src/c_kernel.hpp"
#include "../src/c_selection_impl.hpp"
#include "results.hpp"

#include <Eigen/Dense>

using namespace dpp;

class SelectorsTest: public ::testing::Test {
  const i64 dim = 6;
  std::vector<double> data_;
  i64 row_cnt;

  std::unique_ptr<l_kernel_impl<double>> lkern_;
  std::unique_ptr<c_kernel_impl<double>> ckern_;

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

  std::unique_ptr<l_kernel_impl<double>>& lkern() {
    if (!lkern_) {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mdata{data_.data(), row_cnt, dim};
      Eigen::MatrixXd m = mdata * mdata.adjoint();

      EXPECT_EQ(row_cnt, m.rows());
      EXPECT_EQ(row_cnt, m.cols());

      lkern_ = make_unique<l_kernel_impl<double>>();
      lkern_->init_from_kernel(m.data(), m.rows(), m.cols());
      lkern_->decompose();
    }
    return lkern_;
  }

  std::unique_ptr<c_kernel_impl<double>>& ckern() {
    if (!ckern_) {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mdata{data_.data(), row_cnt, dim};
      typename c_kernel_impl<double>::matrix_t m = mdata;
      ckern_ = make_unique<c_kernel_impl<double>>(std::move(m));
      ckern_->decompose();
    }
    return ckern_;
  }

  i64 rows() const { return row_cnt; }
};

TEST_F(SelectorsTest, SingletonMarginalsAreSame) {
  auto lsel = lkern()->selector();
  auto csel = ckern()->selector();

  for(i64 i = 0; i < rows(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_NEAR(lsel->diagonal_item(i), csel->diagonal_item(i), 1e-6);
  }
}

TEST_F(SelectorsTest, SomeMatricesAreSame) {
  auto lsel = lkern()->selector();
  auto csel = ckern()->selector();

  std::vector<i64> sel{1,5,6};
  auto rw = wrap_result(sel);

  typename l_kernel_selector_impl<double>::matrix_cache_t matl(3, 3);
  typename c_kernel_selector_impl<double>::matrix_cache_t matc(3, 3);

  csel->precompute(rw);

  lsel->fill_cache(rw, matl);
  csel->fill_cache(rw, matc);

  EXPECT_NEAR(matl(0, 0), matc(0, 0), 1e-6);
  EXPECT_NEAR(matl(0, 1), matc(0, 1), 1e-6);
  EXPECT_NEAR(matl(0, 2), matc(0, 2), 1e-6);
  EXPECT_NEAR(matl(1, 2), matc(1, 2), 1e-6);
}


TEST_F(SelectorsTest, SameSelections) {
  auto lsel = lkern()->selector();
  auto csel = lkern()->selector();

  std::vector<i64> lit;
  std::vector<i64> cit;

  auto lw = wrap_result(lit);
  auto cw = wrap_result(cit);

  auto check = [&](i64 size) {
    SCOPED_TRACE(size);
    lit.clear();
    cit.clear();

    lsel->greedy_selection(lw, size);
    csel->greedy_selection(cw, size);

    EXPECT_EQ(lit, cit);
  };

  check(1);
  check(2);
  check(3);
  check(4);
  check(5);
  check(6);
}