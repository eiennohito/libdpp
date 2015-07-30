//
// Created by Arseny Tolmachev on 2015/07/30.
//

#include "common.hpp"
#include "c_selector.hpp"
#include "selector_base.hpp"
#include "c_kernel.hpp"


namespace dpp {

template <typename Fp>
class c_selector_impl :
  public selector_impl_base<c_selector_impl<Fp>, Fp>,
  public tracer_ref_holder<Fp> {

  typedef typename c_kernel_impl<Fp>::kernel_t kernel_t;
  typedef typename c_kernel_impl<Fp>::matrix_t matrix_t;
  typedef typename selector_impl_base<l_kernel_selector_impl<Fp>, Fp>::matrix_cache_t matrix_cache_t;
  typedef typename selector_impl_base<l_kernel_selector_impl<Fp>, Fp>::vector_t vector_t;

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
    if (cache_.rows() < items) {
      items = static_cast<i64>(items * 4 / 3 + 3);
      cache_.conservativeResize(num_items(), items);
    }
  }


public:
  c_selector_impl(const matrix_t &mod_matrix): mod_matrix_(mod_matrix) {
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

template <typename Fp>
void c_selector<Fp>::greedy_max_subset(i64 size, result_holder &out) {
  impl_->greedy_selection(out, size);
}

template <typename Fp>
c_selector<Fp>::~c_selector() {

}

template <typename Fp>
std::unique_ptr<c_selector<Fp>> c_kernel<Fp>::selector() {
  auto &&mat = impl_->kernalized_matrix();
  auto impl = make_unique<c_selector_impl<Fp>>(mat);
  return make_unique<c_selector<Fp>>(std::move(impl));
}

template std::unique_ptr<c_selector<double>> c_kernel<double>::selector();
template std::unique_ptr<c_selector<float>> c_kernel<float>::selector();

LIBDPP_SPECIALIZE_CLASS_FLOATS(c_selector);
LIBDPP_SPECIALIZE_CLASS_FLOATS(c_selector_impl)

}


