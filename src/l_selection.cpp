//
// Created by Arseny Tolmachev on 2015/07/30.
//

#include "selector_base.hpp"
#include "common.hpp"
#include "l_kernel.hpp"
#include "l_selection_impl.hpp"

namespace  dpp {

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

LIBDPP_SPECIALIZE_CLASS_FLOATS(l_kernel_selector_impl)
LIBDPP_SPECIALIZE_CLASS_FLOATS(l_selector)

}
