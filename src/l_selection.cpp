//
// Created by Arseny Tolmachev on 2015/07/30.
//

#include "selector_base.hpp"
#include "common.hpp"
#include "l_kernel.hpp"

namespace  dpp {
template<typename Fp>
class l_kernel_selector_impl :
    public selector_impl_base<l_kernel_selector_impl<Fp>, Fp>,
    public tracer_ref_holder<Fp> {

private:
  typedef typename l_kernel_impl<Fp>::kernel_t kernel_t;
  typedef typename selector_impl_base<l_kernel_selector_impl<Fp>, Fp>::matrix_cache_t matrix_cache_t;
  typedef typename selector_impl_base<l_kernel_selector_impl<Fp>, Fp>::vector_t vector_t;
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

template <typename Fp>
std::unique_ptr<l_kernel_selector_impl<Fp>> l_kernel_impl<Fp>::selector() const {
  return make_unique<l_kernel_selector_impl<Fp>>(this->marginal_kernel_);
}

template <typename Fp>
void l_selector<Fp>::greedy_max_subset(i64 size, result_holder &out) {
  impl_->greedy_selection(out, size);
}

template <typename Fp>
l_selector<Fp>::l_selector(std::unique_ptr<l_kernel_selector_impl<Fp>> &&impl): impl_(std::move(impl)) {
}

template <typename Fp>
l_selector<Fp>::~l_selector() {
}

template <typename Fp>
std::unique_ptr<l_selector<Fp>> l_kernel<Fp>::selector() {
  return make_unique<l_selector<Fp>>(impl_->selector());
}

template std::unique_ptr<l_selector<double>> l_kernel<double>::selector();
template std::unique_ptr<l_selector<float>> l_kernel<float>::selector();

LIBDPP_SPECIALIZE_CLASS_FLOATS(l_kernel_selector_impl)
LIBDPP_SPECIALIZE_CLASS_FLOATS(l_selector)

}
