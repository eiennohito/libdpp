//
// Created by Arseny Tolmachev on 2015/07/28.
//

#include "results.hpp"
#include "common.hpp"

namespace dpp {

i64 *vector_ref_result_holder::data() const {
  return const_cast<i64*>(ref_.data());
}

i64 vector_ref_result_holder::size() const {
  return (i64) ref_.size();
}

void vector_ref_result_holder::append(i64 item) {
  ref_.push_back(item);
}

i64 &result_holder::operator[](i64 idx) {
  DPP_ASSERT(idx >= 0);
  DPP_ASSERT(idx < size());

  return *(data() + idx);
}

bool result_holder::contains(i64 idx) const {
  auto last = data() + size();
  return std::find(data(), last, idx) != last;
}

i64 *memory_area_result_holder::data() const {
  return this->memory_;
}

i64 memory_area_result_holder::size() const {
  return used_;
}

void memory_area_result_holder::append(i64 item) {
  DPP_ASSERT(used_ < max_);
  memory_[used_] = item;
  used_ += 1;
}

const i64 &result_holder::operator[](i64 idx) const {
  DPP_ASSERT(idx >= 0);
  DPP_ASSERT(idx < size());

  return *(data() + idx);
}
}