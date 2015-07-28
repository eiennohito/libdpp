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
 * Assumes presense of
 */
template <typename Derived, typename Fp>
class selector_impl_base: public base_object<Derived>, public trace_support<Fp> {

  typedef Fp fp_t;
  typedef typename eigen_typedefs<fp_t>::matrix_colmajor matrix_cache_t;
  typedef typename eigen_typedefs<fp_t>::vector vector_t;
protected:
  virtual void fill_cache(const std::vector<i64>& indices, matrix_cache_t& mat) = 0;
  virtual Fp diagonal_item(i64 pos) = 0;
  virtual Fp fill_vector(i64 pos, const result_holder& idxs, vector_t& out) = 0;
  virtual i64 num_items() const = 0;

private:
  void trace_vector(const vector_t& vec) {
    trace(vec.data(), vec.size(), TraceType::ProbabilityDistribution);
  }

public:
  void greedy_selection(result_holder& indices, i64 maxSel) {
    typedef typename eigen_typedefs<Fp>::vector vector_t;

    DPP_ASSERT(maxSel < derived().num_items());

    if (maxSel < 1) {
      return;
    }

    auto size = derived().num_items();

    vector_t last(size);

    auto maxProb = std::numeric_limits<fp_t>::lowest();
    i64 selection = 0;

    for (i64 i = 0; i < size; ++i) {
      auto val = derived().diagonal_item(i);
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

      derived().fill_cache(indices, cache);

      decomposition.compute(cache);

      auto Adet = decomposition.vectorD().prod();

      trial.resize(selectionSize);

      maxProb = -50000;

      for (i64 idx = 0; idx < size; ++idx) {
        if (indices.contains(idx)) {
          continue;
        }

        last_item = derived().fill_vector(idx, indices, trial);

        solution = decomposition.solve(trial);

        //cheating on determinants
        auto marginal = Adet * (last_item - trial.dot(solution));

        auto prob = marginal - Adet;
        last(idx) = prob;

        if (prob > maxProb) {
          maxProb = prob;
          selection = idx;
        }
      }

      trace_vector(last);
      last(selection) = 0;
      indices.append(selection);
    }
  }
};

}

#endif //LIBDPP_SELECTOR_BASE_HPP
