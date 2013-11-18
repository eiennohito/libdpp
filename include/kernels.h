#ifndef DPP_KERNEL_H
#define DPP_KERNEL_H 1

#include <vector>
#include <memory>

typedef long long i64;

//#include <boost/preprocessor.hpp>

namespace dpp {

template <typename Fp> class l_kernel_impl;
template <typename Fp> class sampling_subspace_impl;
template <typename Fp> class sampling_subspace;
template <typename Fp> class c_kernel_impl;
template <typename Fp> class dual_sampling_subspace;
template <typename Fp> class c_kernel;
template <typename Fp> class c_kernel_builder_impl;
template <typename Fp> class c_sampler_impl;

enum class TraceType {
  SelectionVector,
  ProbabilityDistribution
};

template <typename Fp> class tracer {
public:
  virtual void trace(Fp *data, i64 count, TraceType type) = 0;
  virtual ~tracer() {}
};

template <typename Fp> class l_kernel {
  typedef typename dpp::l_kernel_impl<Fp> impl_t;
  typedef typename std::unique_ptr<impl_t> impl_ptr;

  impl_ptr impl_;

  l_kernel(impl_ptr &&impl) : impl_{ std::move(impl) } {}

public:
  static l_kernel *from_array(Fp *data, i64 size);

  sampling_subspace<Fp>* sampler();

  sampling_subspace<Fp>* sampler(i64 k);

  ~l_kernel();
};

template <typename Fp> class sampling_subspace {
  std::unique_ptr<sampling_subspace_impl<Fp> > impl_;

public:
  sampling_subspace(std::unique_ptr<sampling_subspace_impl<Fp> > &&impl);

  void register_tracer(tracer<Fp> *ptr);

  std::vector<i64> sample();

  sampling_subspace(sampling_subspace &&o);

  ~sampling_subspace();
};

template <typename Fp> class c_kernel_builder {
  std::unique_ptr<c_kernel_builder_impl<Fp> > impl_;

public:
  c_kernel_builder(i64 from_dim, i64 to_dim);
  c_kernel<Fp> *build_kernel();
  void append(Fp *data, i64 *indices, i64 size);
  void hint_size(i64 size);
  ~c_kernel_builder();
};

template <typename Fp> class c_kernel {
  std::unique_ptr<c_kernel_impl<Fp> > impl_;

public:
  c_kernel(c_kernel_impl<Fp> *impl) : impl_{ impl } {}

  dual_sampling_subspace<Fp> *sampler();
  dual_sampling_subspace<Fp> *sampler(i64 k);

  ~c_kernel();
};

template <typename Fp> class dual_sampling_subspace {
  std::unique_ptr<c_sampler_impl<Fp> > impl_;

public:
  dual_sampling_subspace(std::unique_ptr<c_sampler_impl<Fp> > &&impl)
      : impl_{ std::move(impl) } {}

  void sample(std::vector<i64> &res);
  std::vector<i64> sample();

  void register_tracer(tracer<Fp> *t);

  ~dual_sampling_subspace();
};

extern template class l_kernel<float>;
extern template class l_kernel<double>;
}

#endif