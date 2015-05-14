#include "common.hpp"
#include <limits>
#include <cmath>

namespace dpp {

template <typename Fp>
class l_kernel_impl : public base_kernel<l_kernel_impl<Fp>, Fp> {

  typedef typename eigen_typedefs<Fp>::matrix_colmajor kernel_t;
  // typedef typename eigen_typedefs<Fp>::vector vector_t;

  std::unique_ptr<kernel_t> kernel_;

public:
  kernel_t &kernel() { return *kernel_; }
  const kernel_t &kernel() const { return *kernel_; }

  void init_from_kernel(Fp *data, int rows, int cols) {
    typedef typename eigen_typedefs<Fp>::matrix_rowmajor outer_t;
    Eigen::Map<outer_t> outer(data, rows, cols);
    kernel_ = make_unique<kernel_t>(rows, cols);
    *kernel_ = outer;
  }

  sampling_subspace_impl<Fp> *sampler() const;

  sampling_subspace_impl<Fp> *sampler(i64 k);

  sampling_subspace_impl<Fp> *sampler_greedy(i64 k);


  Fp selection_log_probability(const std::vector<i64> &indices) const {

    //1. create a reduction kernel
    kernel_t reduced(indices.size(), indices.size());

    auto sz = indices.size();

    for (i64 i = 0; i < sz; ++i) {
      for (i64 j = 0; j < sz; ++j) {
        reduced(i, j) = kernel()(indices[i], indices[j]);
      }
    }

    //2. return result
    return std::log(reduced.determinant()) - this->normalizer();
  }
};

template <typename Fp>
class sampling_subspace_impl
    : public base_sampler_impl<sampling_subspace_impl<Fp>, Fp> {

  typedef typename eigen_typedefs<Fp>::matrix_rowmajor subspace_t;

  subspace_t subspace_;
  const l_kernel_impl<Fp> *kernel_;
  const std::vector<i64> vec_indices_;

public:
  sampling_subspace_impl(const l_kernel_impl<Fp> *kernel,
                         std::vector<i64> &&idxs)
      : kernel_{kernel}, vec_indices_{std::move(idxs)} {
    subspace_.resize(vec_indices_.size(), kernel_->cols());
    // reset();
  }

  subspace_t &subspace() { return subspace_; }

  void reset() {
    i64 vecs = vec_indices_.size();
    for (i64 i = 0; i < vecs; ++i) {
      subspace_.row(i) = kernel_->eigenvector(vec_indices_[i]);
    }
  }

private:
  typedef typename Eigen::Matrix<Fp, 1, Eigen::Dynamic> vector_t;
  vector_t cached_;

public:
  /***
   Compute a subspace that is orthogonal to the i-th basis vector.
   The i-th basis vector has all zeros except only one in the i-th component.

   This function computes a projection of e_i to the subspace spanned by the
   vectors,
   then does one step of the Gram-Shmidt process modifying the subspace to be
   orthogonal to e_i.
  */
  void orthogonal_subspace(i64 comp) {
    i64 height = subspace_.rows();
    // compute a projection of a basis vector e_{comp} to the subspace
    cached_ = vector_t::Zero(subspace_.cols());
    for (i64 i = 0; i < height; ++i) {
      cached_ += subspace_.row(i) * subspace_(i, comp);
    }

    cached_.normalize();

    this->trace(cached_.data(), cached_.size(), TraceType::SelectionVector);

    for (i64 i = 0; i < height; ++i) {
      auto &&row = subspace_.row(i);
      auto proj = row.dot(cached_);
      row -= cached_ * proj;
    }
  }

  void sample(std::vector<i64> &buffer, bool greedy) {
    i64 height = subspace_.rows();
    buffer.reserve(height);

    for (i64 i = 0; i < height; ++i) {
#ifdef DPP_TRACE_SAMPLE
      std::cout << "Initial subspace\n" << subspace_ << "\n";
#endif

      cached_ = subspace_.colwise().squaredNorm();  // calculate probabilities

#ifdef DPP_TRACE_SAMPLE
      std::cout << "weights for the selection are:\n" << cached_ << "\n";
#endif

      this->trace(cached_.data(), cached_.size(),
                  TraceType::ProbabilityDistribution);


      i64 selected = 0;

      if (greedy) {
        selected = *std::max_element(cached_.data(), cached_.data() + cached_.size());
      } else {
        auto len = cached_.sum();
        std::uniform_real_distribution<Fp> distr{0, len};
        auto prob = distr(kernel_->rng_);
        Fp val = 0;
        auto total = cached_.size();
        for (; selected < total; ++selected) {
          val += cached_[selected];
          if (val > prob) break;
        }
      }

      buffer.push_back(selected);
      orthogonal_subspace(selected);
#ifdef DPP_TRACE_SAMPLE
      std::cout << "Subspace: removed " << selected << "-th component\n"
                << subspace_ << "\n";
#endif
      this->gram_shmidt_orhonormailze();
    }
  }
};

template <typename Fp>
sampling_subspace_impl<Fp> *l_kernel_impl<Fp>::sampler() const {
  return new sampling_subspace_impl<Fp>{this, this->random_subspace_indices()};
}

template <typename Fp>
sampling_subspace_impl<Fp> *l_kernel_impl<Fp>::sampler(i64 k) {
  return new sampling_subspace_impl<Fp>{this,
                                        this->k_random_subspace_indices(k)};
}

template <typename Fp>
sampling_subspace_impl<Fp> *l_kernel_impl<Fp>::sampler_greedy(i64 k) {
  return new sampling_subspace_impl<Fp>{this,
                                        greedy_basis_indices(k, this->kernel().rows())};
}

template <typename Fp>
void sampling_subspace<Fp>::register_tracer(tracer<Fp> *ptr) {
  impl_->register_tracer(ptr);
}

template <typename Fp>
l_kernel<Fp> *l_kernel<Fp>::from_array(Fp *data, i64 size) {
  auto impl = make_unique<typename l_kernel<Fp>::impl_t>();
  impl->init_from_kernel(data, size, size);
  impl->decompose();
  return new l_kernel<Fp>(std::move(impl));
}

template <typename Fp>
l_kernel<Fp>::~l_kernel<Fp>() {}

template <typename Fp>
sampling_subspace<Fp> *l_kernel<Fp>::sampler() {
  auto impl = impl_->sampler();
  return new sampling_subspace<Fp>{make_unique(impl)};
}

template <typename Fp>
sampling_subspace<Fp> *l_kernel<Fp>::sampler(i64 k) {
  auto impl = impl_->sampler(k);
  return new sampling_subspace<Fp>{make_unique(impl)};
}

template <typename Fp>
sampling_subspace<Fp> *l_kernel<Fp>::sampler_greedy(i64 k) {
  auto impl = impl_->sampler_greedy(k);
  return new sampling_subspace<Fp>{make_unique(impl)};
}

template <typename Fp>
sampling_subspace<Fp>::sampling_subspace(
    std::unique_ptr<sampling_subspace_impl<Fp> > &&impl)
    : impl_{std::move(impl)} {}

template <typename Fp>
sampling_subspace<Fp>::sampling_subspace(sampling_subspace<Fp> &&o)
    : impl_{std::move(o.impl_)} {}

template <typename Fp>
std::vector<i64> sampling_subspace<Fp>::sample() {
  std::vector<i64> vec;
  this->sample(vec);
  return std::move(vec);
}

template <typename Fp>
i64 sampling_subspace<Fp>::sample(std::vector<i64> &out) {
  impl_->reset();
  impl_->sample(out, false);
  return 0;
}

template <typename Fp>
i64 sampling_subspace<Fp>::greedy(std::vector<i64> &out) {
  impl_->reset();
  impl_->sample(out, true);
  return 0;
}

template <typename Fp>
Fp l_kernel<Fp>::selection_log_probability(std::vector<i64> &indices) {
  return this->impl_->selection_log_probability(indices);
}

template <typename Fp>
sampling_subspace<Fp>::~sampling_subspace<Fp>() {}


//explicit template instantiation

template class l_kernel<float>;
template class l_kernel<double>;

template class sampling_subspace<float>;
template class sampling_subspace<double>;

}



