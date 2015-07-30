//
// Created by Arseny Tolmachev on 2015/07/30.
//

#ifndef LIBDPP_L_SELECTION_HPP
#define LIBDPP_L_SELECTION_HPP

namespace dpp {

template <typename Fp>
class l_kernel_selector_impl;

template <typename Fp>
class l_selector {
private:
  std::unique_ptr<l_kernel_selector_impl<Fp>> impl_;

public:
  l_selector(std::unique_ptr<l_kernel_selector_impl<Fp>> &&impl_);

public:
  void greedy_max_subset(i64 size, result_holder& out);

  ~l_selector();
};

}

#endif //LIBDPP_L_SELECTION_HPP
