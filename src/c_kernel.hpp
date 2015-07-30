//
// Created by Arseny Tolmachev on 2015/07/30.
//

#ifndef LIBDPP_C_KERNEL_HPP
#define LIBDPP_C_KERNEL_HPP

#include "common.hpp"
#include "c_selector.hpp"

namespace dpp {

template <typename Fp>
class c_kernel_impl : public base_kernel<c_kernel_impl<Fp>, Fp> {
public:
  typedef typename Eigen::Matrix<Fp, Eigen::Dynamic, Eigen::Dynamic,
      Eigen::RowMajor> matrix_t;
  typedef typename eigen_typedefs<Fp>::matrix_colmajor kernel_t;

private:
  // dual DPP-kernel, has dimensions of D \times D
  kernel_t kernel_;

  // dense matrix of items, each row contains
  // vector similarity features (norm == 1) multiplied
  // by scalar quality features, so the norm == quality.
  // Dimensions of matrix are N \times D
  matrix_t matrix_;

  //dense matrix of items for easy calculation of marginal kernel elements
  matrix_t kernalized_;

public:
  c_kernel_impl(matrix_t &&matrix) : matrix_{std::move(matrix)} {
    kernel_ = matrix_.adjoint() * matrix_;
  }

  kernel_t &kernel() { return kernel_; }
  const kernel_t &kernel() const { return kernel_; }

  matrix_t &matrix() { return matrix_; }
  const matrix_t &matrix() const { return matrix_; }

  Fp selection_log_probability(const std::vector<i64> &indices) const;

  matrix_t& kernalized_matrix();
};

}

#endif //LIBDPP_C_KERNEL_HPP
