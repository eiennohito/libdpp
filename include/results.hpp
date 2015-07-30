//
// Created by Arseny Tolmachev on 2015/07/28.
//

#ifndef LIBDPP_RESULTS_HPP
#define LIBDPP_RESULTS_HPP


#include "fwddefs.hpp"

#include <vector>

namespace dpp {

class result_holder {
public:
  virtual i64* data() const = 0;
  virtual i64 size() const = 0;
  virtual void append(i64 item) = 0;

  i64 & operator[](i64 idx);
  const i64 & operator[](i64 idx) const;

  bool contains(i64 idx) const;

  virtual ~result_holder() {}
};


class vector_ref_result_holder : public result_holder {
private:
  std::vector<i64>& ref_;
public:
  vector_ref_result_holder(std::vector<i64> &ref) : ref_(ref) { }

  virtual i64 *data() const override;
  virtual i64 size() const override;
  virtual void append(i64 item) override;
};

class memory_area_result_holder: public result_holder {
  i64 *memory_;
  i64 used_;
  i64 max_;

public:
  memory_area_result_holder(i64 *memory_, i64 max) : memory_(memory_), max_(max), used_(0) { }

  virtual i64 *data() const override;
  virtual i64 size() const override;
  virtual void append(i64 item) override;
};

inline vector_ref_result_holder wrap_result(std::vector<i64>& ref) {
  return vector_ref_result_holder{ref};
}

}

#endif //LIBDPP_RESULTS_HPP
