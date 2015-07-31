//
// Created by Arseny Tolmachev on 2015/07/30.
//

#include "common.hpp"
#include "selector_base.hpp"
#include "c_selection_impl.hpp"


namespace dpp {

template <typename Fp>
void c_selector<Fp>::greedy_max_subset(i64 size, result_holder &out) {
  impl_->greedy_selection(out, size);
}

template <typename Fp>
c_selector<Fp>::~c_selector() {

}

LIBDPP_SPECIALIZE_CLASS_FLOATS(c_selector);
LIBDPP_SPECIALIZE_CLASS_FLOATS(c_kernel_selector_impl)

}


