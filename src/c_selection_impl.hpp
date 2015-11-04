//
// Created by Arseny Tolmachev on 2015/07/31.
//

#ifndef LIBDPP_C_SELECTION_IMPL_HPP
#define LIBDPP_C_SELECTION_IMPL_HPP

#include "common.hpp"
#include "selector_base.hpp"
#include "c_kernel.hpp"

namespace dpp {

template <typename Fp>
class c_kernel_selector_impl :
    public selector_impl_base<c_kernel_selector_impl<Fp>, Fp>,
    public tracer_ref_holder<Fp> {

public:
  typedef c_mat_kernel_t<Fp> kernel_t;
  typedef c_mat_matrix_t<Fp> matrix_t;
  typedef typename selector_impl_base<l_kernel_selector_impl<Fp>, Fp>::matrix_cache_t matrix_cache_t;
  typedef typename selector_impl_base<l_kernel_selector_impl<Fp>, Fp>::vector_t vector_t;

private:
  matrix_t cache_;
  const matrix_t& mod_matrix_;
  i64 computed_ = 0;

private:
  void computeDiagonal() {
    ensureCols(1);

    auto sz = num_items();

    for (i64 i = 0; i < sz; ++i) {
      auto&& row = mod_matrix_.row(i);
      cache_(i, 0) = row.dot(row);
    }
  }

  void ensureCols(i64 items) {
    if (cache_.cols() < items) {
      items = static_cast<i64>(items * 4 / 3 + 3);
      cache_.conservativeResize(num_items(), items);
    }
  }


public:
  c_kernel_selector_impl(const matrix_t &mod_matrix): mod_matrix_(mod_matrix) {
    ensureCols(2);
    computeDiagonal();
  }

  virtual void reset() override {
    computed_ = 0;
  }

  virtual void fill_cache(const result_holder &indices, matrix_cache_t &mat) {
    i64 sz = indices.size();

    for (i64 i = 0; i < sz; ++i) {
      mat(i, i) = cache_(indices[i], 0);
      for (i64 j = i; j < sz; ++j) {
        auto val = cache_(indices[j], i + 1);
        mat(i, j) = val;
        mat(j, i) = val;
      }
    }
  }


  virtual void hintSize(i64 maxSelection) override {
    ensureCols(maxSelection + 1);
  }

  virtual Fp diagonal_item(i64 pos) {
    return cache_(pos, 0);
  }

  virtual Fp fill_vector(i64 pos, const result_holder &idxs, vector_t &out) {
    auto sz = idxs.size();
    out = cache_.row(pos).tail(sz);
    return cache_(pos, 0);
  }

  virtual i64 num_items() const {
    return mod_matrix_.rows();
  }


  virtual void precompute(const result_holder &idxs) override {
    ensureCols(idxs.size() + 1);
    auto items = num_items();
    auto offdiag = idxs.size();

    for (i64 item = 0; item < items; ++item) {
      auto &&r1 = mod_matrix_.row(item);
      for (i64 i = computed_; i < offdiag; ++i) {
        auto &&r2 = mod_matrix_.row(idxs[i]);
        cache_(item, i + 1) = r1.dot(r2);
      }
    }

    computed_ = offdiag;
  }
};

}

#endif //LIBDPP_C_SELECTION_IMPL_HPP
