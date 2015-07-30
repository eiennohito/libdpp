//
// Created by Arseny Tolmachev on 2015/07/30.
//

#ifndef LIBDPP_C_SELECTION_HPP
#define LIBDPP_C_SELECTION_HPP


#include <memory>

namespace dpp {

template <typename Fp>
class c_selector_impl;

template <typename Fp>
class c_selector {
  std::unique_ptr<c_selector_impl<Fp>> impl_;

public:
  c_selector(std::unique_ptr<c_selector_impl<Fp>> &&impl) : impl_(std::move(impl)) {}

  void greedy_max_subset(i64 size, result_holder& out);

  virtual ~c_selector();
};

}


#endif //LIBDPP_C_SELECTION_HPP
