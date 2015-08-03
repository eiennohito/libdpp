//
// Created by Arseny Tolmachev on 2015/08/03.
//

#ifndef LIBDPP_PERF_COMMON_HPP
#define LIBDPP_PERF_COMMON_HPP

#include <chrono>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>

template <typename Fn>
auto measure(Fn fn) -> std::pair<decltype(fn()), std::chrono::high_resolution_clock::duration> {
  auto begin = std::chrono::high_resolution_clock::now();
  auto res = fn();
  auto end = std::chrono::high_resolution_clock::now();
  return std::make_pair(res, end - begin);
}

struct measure_stats {
  double avg;
  double max;
  double min;
  double variance;
};

template <typename Fn>
class measurer {
  Fn fn;
  typedef decltype(fn()) fun_result;

  std::vector<double> millis_;
  std::vector<fun_result> results_;

  typedef std::chrono::duration<double, std::milli> dmillis;

public:
  measurer(Fn fn) : fn(fn) { }

  void reset() {
    millis_.clear();
    results_.clear();
  }

  measure_stats stat() const {
    if (millis_.empty()) {
      return measure_stats{0};
    }

    double mxv = 0;
    double miv = std::numeric_limits<double>::max();
    double mean = 0;
    double m2 = 0;
    int n = 0;

    for (auto &&x: millis_) {
      n += 1;
      auto delta = x - mean;
      mean += delta / n;
      m2 += delta * (x - mean);

      if (x > mxv) {
        mxv = x;
      }

      if (x < miv) {
        miv = x;
      }
    }

    return measure_stats {
        mean, mxv, miv, m2 / (n - 1)
    };
  }

  measure_stats run(int times) {
    for (int i = 0; i < times; ++i) {
      auto res = measure(fn);
      millis_.push_back(std::chrono::duration_cast<dmillis>(res.second).count());
      results_.push_back(res.first);
    }
    return stat();
  }
};

template <typename Fn>
measurer<Fn> make_measurer(Fn fn) {
  return measurer<Fn>{fn};
}



#endif //LIBDPP_PERF_COMMON_HPP
