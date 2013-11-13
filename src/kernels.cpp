#include "kernels.h"

// use Eigen's dense matrices
#include "Dense"
//#include "Map.h"

// use Eigen's eigendecomposition
#include <Eigenvalues>

#include <random>
#include <vector>

#include <iostream>

#ifndef NDEBUG
#define DPP_ASSERT(x) assert(x)
#else
#define DPP_ASSERT(x)
#endif


namespace dpp {

template <typename R, typename... T>
std::unique_ptr<R> make_unique(T &&... vals) {
  return std::unique_ptr<R>(new R(std::forward<T>(vals)...));
}

template <typename R> std::unique_ptr<R> make_unique(R *ptr) {
  return std::unique_ptr<R>(ptr);
}

template <typename Fp> struct eigen_typedefs {
  typedef Eigen::Matrix<Fp, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  matrix_rowmajor;
  typedef Eigen::Matrix<Fp, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
  matrix_colmajor;

  typedef Eigen::Matrix<Fp, Eigen::Dynamic, 1> vector;
};

template <typename Fp> class l_kernel_impl {

  typedef typename eigen_typedefs<Fp>::matrix_colmajor kernel_t;
  typedef typename eigen_typedefs<Fp>::vector vector_t;
  std::unique_ptr<kernel_t> kernel_;
  Eigen::SelfAdjointEigenSolver<kernel_t> eigen_;
  
  std::unique_ptr<kernel_t> polynomials_;

public:
  mutable std::mt19937 rng_{ std::random_device()() };

  void init_from_kernel(Fp *data, int rows, int cols) {
    typedef typename eigen_typedefs<Fp>::matrix_rowmajor outer_t;
    Eigen::Map<outer_t> outer(data, rows, cols);
    kernel_ = make_unique<kernel_t>(rows, cols);
    *kernel_ = outer;
  }

  void decompose() { eigen_.compute(*kernel_); }

  std::vector<i64> random_subspace_indices() const {
    std::vector<i64> vec;
    vec.reserve(static_cast<i64>(sqrt(kernel_->rows()))); // reserve some memory
    auto &eigen_values = eigen_.eigenvalues().array();

    std::uniform_real_distribution<Fp> distr{ 0, 1 };
    for (i64 i = 0; i < eigen_values.cols(); ++i) {
      auto rn = distr(rng_);
      auto ev = eigen_values(i);
      auto prob = ev / (ev + 1);
      if (rn < prob) {
        vec.push_back(i);
      }
    }

    return std::move(vec);
  }
  
  /***
   To create a subsample k-DPP basis we need to compute
   elementary symmetric polynomials.
   
   They are written to the (N + 1) \times (k + 1) matrix,
   where N is the dimension of the kernel.
   
   This matrix is NOT 0-based, but 1-based, zeros are padding.
   */
  void compute_symmetric_polynomials(i64 k) {
    if (!polynomials_) {
      polynomials_ = make_unique<kernel_t>();
    }
    kernel_t& pols = *polynomials_;
    
    if (pols.cols() >= (k + 1)) { return; }
    
    auto &ev = eigen_.eigenvalues();
    auto nrow = kernel_->rows();
    pols.resize(nrow + 1, k + 1);
    pols.row(0).setConstant(0);
    pols.col(0).setConstant(1);
    for (i64 l = 1; l <= k; ++l) {
      for (i64 n = 1; n <= nrow; ++n) {
        pols(n, l) = pols(n - 1, l) + ev(n - 1) * pols(n - 1, l - 1);
      }
    }
  }
  
  //non-const because of the need to precompute
  //symmetric polynomials
  std::vector<i64> k_random_subspace_indices(i64 k) {
    compute_symmetric_polynomials(k);
    
    std::vector<i64> res;
    res.reserve(k);
    
    std::uniform_real_distribution<Fp> distr;
    
    auto l = k;
    auto &ev = eigen_.eigenvalues();
    auto &ep = *polynomials_; //1-based matrix
    auto N = ev.size();
    for (i64 n = N - 1; n >= 0; --n) { //because of zero-based indices n is less by 1
      if (l == 0) break;
      auto rv = distr(rng_);
      // n-1, l-1 | n,l ; l is 1-based here, n is 0-based
      auto prob = ev(n) * ep(n, l - 1) / ep(n + 1, l);
      if (rv < prob) {
        res.push_back(n);
        --l;
      }
    }
    
    return std::move(res);
  }

  auto eigenvector(i64 pos) const -> decltype(eigen_.eigenvectors().row(pos)) {
    return eigen_.eigenvectors().row(pos);
  }

  i64 cols() const { return kernel_->cols(); }

  sampling_subspace_impl<Fp> *sampler() const;
  
  sampling_subspace_impl<Fp> *sampler(i64 k);
};

template <typename Fp>
l_kernel<Fp> *l_kernel<Fp>::from_array(Fp *data, i64 size) {
  auto impl = make_unique<typename l_kernel<Fp>::impl_t>();
  impl->init_from_kernel(data, size, size);
  impl->decompose();
  return new l_kernel<Fp>(std::move(impl));
}

template <typename Fp> class sampling_subspace_impl {
  typedef typename eigen_typedefs<Fp>::matrix_rowmajor subspace_t;

  subspace_t subspace_;
  const l_kernel_impl<Fp> *kernel_;
  const std::vector<i64> vec_indices_;
  tracer<Fp>* tracer_;

public:
  sampling_subspace_impl(const l_kernel_impl<Fp> *kernel,
                         std::vector<i64> &&idxs)
      : kernel_{ kernel }, vec_indices_{ std::move(idxs) } {
    subspace_.resize(vec_indices_.size(), kernel_->cols());
    //reset();
  }

  void reset() {
    i64 vecs = vec_indices_.size();
    for (i64 i = 0; i < vecs; ++i) {
      subspace_.row(i) = kernel_->eigenvector(vec_indices_[i]);
    }
  }

  void gram_shmidt_orhonormailze() {
    auto height = subspace_.rows();
    for (i64 row = 0; row < height; ++row) {
      auto &&pivot_row = subspace_.row(row);
      
      auto norm = pivot_row.norm();
      if (norm > 1e-10) {
        pivot_row /= norm;
      } else {
        pivot_row.setZero();
      }
      //pivot_row.normalize();
      
      //DPP_ASSERT(std::abs(subspace_.row(row).norm() - 1) < 1e-15);
      for (i64 other = row + 1; other < height; ++other) {
        auto &&other_row = subspace_.row(other);
        auto projection = other_row.dot(subspace_.row(row));
        subspace_.row(other) -= (pivot_row * projection);
        DPP_ASSERT(std::abs(subspace_.row(row).dot(subspace_.row(other))) < 1e-15);
      }
    }
    
    //subspace_t copy = subspace_;
    Eigen::ColPivHouseholderQR<subspace_t> solver;
    solver.compute(subspace_);
    std::cout << "subspace has rank " << solver.rank() << "\n";
    std::cout << "Our subspace is orthogonal: " << is_orthogonal() << "\n";
  }
  
  bool is_orthogonal() const {
    auto height = subspace_.rows();
    bool result = true;
    for (i64 row = 0; row < height; ++row) {
      for (i64 other = row + 1; other < height; ++other) {
        auto sim = subspace_.row(row).dot(subspace_.row(other));
        if (std::abs(sim) > 1e-12) {
          std::cout << "![" << row << ", " << other << ": " << sim << "] ";
          result = false;
        }
      }
    }
    return result;
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
    
    if (tracer_) {
      tracer_->trace(cached_.data(), cached_.size(), TraceType::SelectionVector);
    }
    
    for (i64 i = 0; i < height; ++i) {
      auto &&row = subspace_.row(i);
      auto proj = row.dot(cached_);
      row -= cached_ * proj;
    }
  }

  void sample(std::vector<i64> &buffer) {
    i64 height = subspace_.rows();
    buffer.reserve(height);

    for (i64 i = 0; i < height; ++i) {
      std::cout << "Initial subspace\n" << subspace_ << "\n";
      
      cached_ = subspace_.colwise().squaredNorm(); // calculate probabilities
      
      std::cout << "weights for the selection are:\n" << cached_ << "\n";
      
      if (tracer_) {
        tracer_->trace(cached_.data(), cached_.size(), TraceType::ProbabilityDistribution);
      }
      
      auto len = cached_.sum();
      std::uniform_real_distribution<Fp> distr{ 0, len };
      auto prob = distr(kernel_->rng_);
      i64 selected = 0;
      Fp val = 0;
      auto total = cached_.size();
      for (; selected < total; ++selected) {
        val += cached_[selected];
        if (val > prob)
          break;
      }

      buffer.push_back(selected);
      orthogonal_subspace(selected);
      std::cout << "Subspace: removed " << selected << "-th component\n" << subspace_ << "\n";
      gram_shmidt_orhonormailze();
    }
  }
  
  void register_tracer(tracer<Fp>* ptr) {
    tracer_ = ptr;
  }
};
  
  
template <typename Fp>
sampling_subspace_impl<Fp> *l_kernel_impl<Fp>::sampler() const {
  return new sampling_subspace_impl<Fp>{ this, random_subspace_indices() };
}
  
  template<typename Fp>
  sampling_subspace_impl<Fp> *l_kernel_impl<Fp>::sampler(i64 k) {
    return new sampling_subspace_impl<Fp>{ this, k_random_subspace_indices(k) };
  }
  
  template<typename Fp>
  void sampling_subspace<Fp>::register_tracer(tracer<Fp> *ptr) {
    impl_->register_tracer(ptr);
  }

template <typename Fp> l_kernel<Fp>::~l_kernel<Fp>() {}

template <typename Fp> sampling_subspace<Fp> l_kernel<Fp>::sampler() {
  auto impl = impl_->sampler();
  return sampling_subspace<Fp>{ make_unique(impl) };
}
  
  template <typename Fp> sampling_subspace<Fp> l_kernel<Fp>::sampler(i64 k) {
    auto impl = impl_->sampler(k);
    return sampling_subspace<Fp>{ make_unique(impl) };
  }

template <typename Fp>
sampling_subspace<Fp>::sampling_subspace(
    std::unique_ptr<sampling_subspace_impl<Fp> > &&impl)
    : impl_{ std::move(impl) } {}

template <typename Fp>
sampling_subspace<Fp>::sampling_subspace(sampling_subspace<Fp> &&o)
    : impl_{ std::move(o.impl_) } {}

template <typename Fp> std::vector<i64> sampling_subspace<Fp>::sample() {
  std::vector<i64> vec;
  impl_->reset();
  impl_->sample(vec);
  return std::move(vec);
}

template <typename Fp> sampling_subspace<Fp>::~sampling_subspace<Fp>() {}

template class l_kernel<float>;
template class l_kernel<double>;

template class sampling_subspace<float>;
template class sampling_subspace<double>;
}