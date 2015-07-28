//
// Created by Arseny Tolmachev on 2015/07/28.
//

#ifndef LIBDPP_FWDDEFS_HPP
#define LIBDPP_FWDDEFS_HPP

typedef long long i64;

namespace dpp {

template <typename Fp>
class l_kernel_impl;
template <typename Fp>
class sampling_subspace_impl;
template <typename Fp>
class sampling_subspace;
template <typename Fp>
class c_kernel_impl;
template <typename Fp>
class dual_sampling_subspace;
template <typename Fp>
class c_kernel;
template <typename Fp>
class c_kernel_builder_impl;
template <typename Fp>
class c_sampler_impl;

enum class TraceType {
  SelectionVector,
  ProbabilityDistribution
};

}

#endif //LIBDPP_FWDDEFS_HPP
