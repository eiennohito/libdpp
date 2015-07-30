//
// Created by Arseny Tolmachev on 2015/07/28.
//

#ifndef LIBDPP_SELECTOR_BASE_HPP
#define LIBDPP_SELECTOR_BASE_HPP

#include <results.hpp>
#include "common.hpp"
#include "../include/kernels.h"

namespace dpp {

/**
 * Assumes presense of function trace(...) as if child was implementing trace_support interface
 */
template <typename Derived, typename Fp>
class selector_impl_base:
    public base_object<Derived> {


  typedef Fp fp_t;
protected:
  typedef typename eigen_typedefs<fp_t>::matrix_colmajor matrix_cache_t;
  typedef typename eigen_typedefs<fp_t>::vector vector_t;

public:
  virtual void fill_cache(const result_holder& indices, matrix_cache_t& mat) = 0;
  virtual Fp diagonal_item(i64 pos) = 0;
  virtual Fp fill_vector(i64 pos, const result_holder& idxs, vector_t& out) = 0;
  virtual i64 num_items() const = 0;
  virtual void precompute(const result_holder& idxs) {}

private:
  void trace_vector(const vector_t& vec) {
    this->derived().trace(vec.data(), vec.size(), TraceType::ProbabilityDistribution);
  }

public:
  void greedy_selection(result_holder& indices, i64 maxSel) {
    typedef typename eigen_typedefs<Fp>::vector vector_t;

    DPP_ASSERT(maxSel < this->derived().num_items());

    if (maxSel < 1) {
      return;
    }

    auto size = this->derived().num_items();

    vector_t last(size);

    auto maxProb = std::numeric_limits<fp_t>::lowest();
    i64 selection = 0;

    for (i64 i = 0; i < size; ++i) {
      auto val = this->derived().diagonal_item(i);
      last(i) = val;
      if (val > maxProb) {
        maxProb = val;
        selection = i;
      }
    }

    trace_vector(last);

    last(selection) = 0;
    indices.append(selection);

    if (maxSel == 1) {
      return;
    }

    matrix_cache_t cache;
    vector_t trial, solution;
    fp_t last_item;

    Eigen::LDLT<matrix_cache_t> decomposition;

    for (i64 selectionSize = 1; selectionSize < maxSel; ++selectionSize) {

      cache.resize(selectionSize, selectionSize);

      this->derived().fill_cache(indices, cache);

      decomposition.compute(cache);

      trial.resize(selectionSize);

      maxProb = std::numeric_limits<fp_t>::lowest();

      for (i64 idx = 0; idx < size; ++idx) {
        if (indices.contains(idx)) {
          continue;
        }

        last_item = this->derived().fill_vector(idx, indices, trial);

        solution = decomposition.solve(trial);

        //cheating on determinants
        auto prob = last_item - trial.dot(solution);

        last(idx) = prob;

        if (prob > maxProb) {
          maxProb = prob;
          selection = idx;
        }
      }

      trace_vector(last);
      last(selection) = 0;
      indices.append(selection);
      this->derived().precompute(indices);
    }
  }
};

}

#endif //LIBDPP_SELECTOR_BASE_HPP
