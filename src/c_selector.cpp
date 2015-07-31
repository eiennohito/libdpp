//
// Created by Arseny Tolmachev on 2015/07/30.
//

#include "common.hpp"
#include "c_selector.hpp"
#include "selector_base.hpp"
#include "c_kernel.hpp"
#include "c_selection_impl.hpp"


namespace dpp {

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


