//
// Created by Arseny Tolmachev on 2015/07/30.
//

#include "selector_base.hpp"
#include "common.hpp"
#include "l_kernel.hpp"
#include "l_selection_impl.hpp"

namespace  dpp {

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
