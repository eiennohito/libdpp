//
// Created by Arseny Tolmachev on 2015/07/31.
//

#ifndef LIBDPP_L_SELECTION_IMPL_HPP
#define LIBDPP_L_SELECTION_IMPL_HPP

#include "common.hpp"
#include "selector_base.hpp"

namespace dpp {

template<typename Fp>
class l_kernel_selector_impl :
    public selector_impl_base<l_kernel_selector_impl<Fp>, Fp>,
    public tracer_ref_holder<Fp> {

public:
  typedef typename l_kernel_impl<Fp>::kernel_t kernel_t;
  typedef typename selector_impl_base<l_kernel_selector_impl<Fp>, Fp>::matrix_cache_t matrix_cache_t;
  typedef typename selector_impl_base<l_kernel_selector_impl<Fp>, Fp>::vector_t vector_t;

private:
  const kernel_t& marginal_;

public:
  l_kernel_selector_impl(const kernel_t &marginal_) : marginal_(marginal_) { }

public:
  virtual void fill_cache(const result_holder &indices, matrix_cache_t &mat) override {
    i64 sz = indices.size();
    for (i64 i = 0; i < sz; ++i) {
      for (i64 j = 0; j < sz; ++j) {
        mat(i,j) = marginal_(indices[i], indices[j]);
      }
    }
  }

  virtual Fp diagonal_item(i64 pos) override {
    return marginal_(pos, pos);
  }

  virtual Fp fill_vector(i64 pos, const result_holder &idxs, vector_t &out) override {
    auto sz = idxs.size();
    for (i64 i = 0; i < sz; ++i) {
      out(i) = marginal_(pos, idxs[i]);
    }
    return marginal_(pos, pos);
  }

  virtual i64 num_items() const override {
    return marginal_.rows();
  }
};
}

#endif //LIBDPP_L_SELECTION_IMPL_HPP
