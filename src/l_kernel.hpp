//
// Created by Arseny Tolmachev on 2015/07/30.
//

#ifndef LIBDPP_L_KERNEL_HPP
#define LIBDPP_L_KERNEL_HPP

#include "l_selection.hpp"

namespace dpp {

template <typename Fp>
class l_kernel_impl : public base_kernel<l_kernel_impl<Fp>, Fp> {
public:

  typedef typename eigen_typedefs<Fp>::matrix_colmajor kernel_t;
  // typedef typename eigen_typedefs<Fp>::vector vector_t;

private:
  std::unique_ptr<kernel_t> kernel_;

  kernel_t marginal_kernel_;

public:
  kernel_t &kernel() { return *kernel_; }
  const kernel_t &kernel() const { return *kernel_; }

  void init_from_kernel(Fp *data, int rows, int cols) {
    typedef typename eigen_typedefs<Fp>::matrix_rowmajor outer_t;
    Eigen::Map<outer_t> outer(data, rows, cols);
    kernel_ = make_unique<kernel_t>(rows, cols);
    *kernel_ = outer;
  }

  sampling_subspace_impl<Fp> *sampler() const;

  sampling_subspace_impl<Fp> *sampler(i64 k);

  sampling_subspace_impl<Fp> *sampler_greedy(i64 k);

  std::unique_ptr<l_kernel_selector_impl<Fp>> selector() const;

  virtual void decompose();

  Fp selection_log_probability(const std::vector<i64> &indices) const {
    //1. create a reduction kernel
    kernel_t reduced(indices.size(), indices.size());

    auto sz = indices.size();

    for (i64 i = 0; i < sz; ++i) {
      for (i64 j = 0; j < sz; ++j) {
        reduced(i, j) = marginal_kernel_(indices[i], indices[j]);
      }
    }

#ifdef DPP_TRACE_KERNELS
    std::cout << "marginal selection kernel (L->K):\n" << reduced << "\n";
#endif //DPP_TRACE_KERNELS

    Eigen::LDLT<kernel_t> ldlt(reduced);

    //2. return result
    return ldlt.vectorD().array().log().sum();
  }

};

}

#endif //LIBDPP_L_KERNEL_HPP
