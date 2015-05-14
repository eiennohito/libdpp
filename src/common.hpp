#include "kernels.h"

// use Eigen's dense matrices
#include "Eigen/Dense"

// use Eigen's eigendecomposition
#include "Eigen/Eigenvalues"

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

template <typename R>
std::unique_ptr<R> make_unique(R *ptr) {
  return std::unique_ptr<R>(ptr);
}

template <typename Fp>
struct eigen_typedefs {
  typedef Eigen::Matrix<Fp, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      matrix_rowmajor;
  typedef Eigen::Matrix<Fp, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      matrix_colmajor;

  typedef Eigen::Matrix<Fp, Eigen::Dynamic, 1> vector;
};

template <typename Derived>
class base_object {
protected:
  Derived &derived() { return static_cast<Derived &>(*this); }
  const Derived &derived() const { return static_cast<const Derived &>(*this); }
};


template <typename Derived, typename Fp>
class base_kernel : public base_object<Derived> {
protected:
  typedef typename eigen_typedefs<Fp>::matrix_colmajor kernel_t;

  Eigen::SelfAdjointEigenSolver<kernel_t> eigen_;
  std::unique_ptr<kernel_t> polynomials_;

  //A normalizer constant for computing subset selection probability
  Fp normalizer_;

  /**
   * Compute a normalizing constant for subset selection probability
   * It is equal to the det(L + I)
   *
   * That probability can become very small, so use the log space for the precision.
   *
   * Because det(L) is equal to sum of its eigenvalues,
   * and non-zero eigenvalues are equal for the L and C kernels,
   * it is possible to use a common function to compute eigenvalues.
   *
   * The final piece for this function is the property of eigendecomposition
   * and determinants so det(A + I) = sum(\lambda_i + 1)
   */
  Fp compute_normalizer() const {
    //compute log(det(L + I))

    auto &ev = this->eigenvalues();
    auto sz = ev.size();
    auto tmp = Fp{0};
    for (i64 i = 0; i < sz; ++i) {
      tmp += std::log(ev(i) + 1);
    }
    return tmp;
  }

public:
  mutable std::mt19937 rng_{std::random_device()()};

  std::vector<i64> random_subspace_indices() const {
    std::vector<i64> vec;
    vec.reserve(static_cast<i64>(
                    sqrt(this->derived().kernel().rows())));  // reserve some memory
    auto &eigen_values = eigenvalues().array();

    std::uniform_real_distribution<Fp> distr{0, 1};
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

  Fp normalizer() const {
    return normalizer_;
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
    kernel_t &pols = *polynomials_;

    if (pols.cols() >= (k + 1)) {
      return;
    }

    auto &ev = this->eigenvalues();
    auto nrow = this->derived().kernel().rows();
    pols.resize(nrow + 1, k + 1);
    pols.row(0).setConstant(0);
    pols.col(0).setConstant(1);
    for (i64 l = 1; l <= k; ++l) {
      for (i64 n = 1; n <= nrow; ++n) {
        pols(n, l) = pols(n - 1, l) + ev(n - 1) * pols(n - 1, l - 1);
      }
    }

#ifdef DPP_TRACE_SAMPLE
    std::cout << "Eigenvalues are: " << ev << "\n";
    std::cout << "Computed elementary polynomials:\n" << pols << "\n";

    kernel_t probs(nrow, k);
    for (i64 l = 0; l < k; ++l) {
      for (i64 n = 0; n < nrow; ++n) {
        probs(n, l) = ev(n) * pols(n, l) / pols(n + 1, l + 1);
      }
    }

    std::cout << "Probabilities to select are:\n" << probs << "\n";
#endif
  }

  // non-const because of the need to precompute
  // symmetric polynomials
  std::vector<i64> k_random_subspace_indices(i64 k) {
    compute_symmetric_polynomials(k);

    std::vector<i64> res;
    res.reserve(k);

    std::uniform_real_distribution<Fp> distr;

    auto l = k;
    auto &ev = this->eigenvalues();
    auto &ep = *polynomials_;  // 1-based matrix
    auto N = ev.size();
    for (i64 n = N - 1; n >= 0;
         --n) {  // because of zero-based indices n is less by 1
      if (l == 0) break;
      auto rv = distr(this->rng_);
      // n-1, l-1 | n,l ; l is 1-based here, n is 0-based
      auto prob = ev(n) * ep(n, l - 1) / ep(n + 1, l);
      if (rv < prob) {
        res.push_back(n);
        --l;
      }
    }

    return std::move(res);
  }

  virtual void decompose() {
    eigen_.compute(this->derived().kernel());
    normalizer_ = compute_normalizer();
  }

  auto eigenvector(i64 pos) const -> decltype(eigen_.eigenvectors().row(pos)) {
    return eigen_.eigenvectors().row(pos);
  }

  auto eigenvectors() const -> decltype(eigen_.eigenvectors()) {
    return eigen_.eigenvectors();
  }

  auto eigenvalues() const -> decltype(eigen_.eigenvalues()) {
    return eigen_.eigenvalues();
  }

  i64 cols() const { return this->derived().kernel().cols(); }

  virtual ~base_kernel() {}
};

template <typename Derived, typename Fp>
class base_sampler_impl : public base_object<Derived> {
public:
  typedef typename eigen_typedefs<Fp>::matrix_rowmajor subspace_t;

protected:
  void gram_shmidt_orhonormailze() {
    subspace_t &subspace = this->derived().subspace();
    auto height = subspace.rows();
    for (i64 row = 0; row < height; ++row) {
      auto &&pivot_row = subspace.row(row);

      auto norm = pivot_row.norm();
      if (norm > 1e-10) {
        pivot_row /= norm;
      } else {
        pivot_row.setZero();
      }
      // pivot_row.normalize();

      // DPP_ASSERT(std::abs(subspace_.row(row).norm() - 1) < 1e-15);
      for (i64 other = row + 1; other < height; ++other) {
        auto &&other_row = subspace.row(other);
        auto projection = other_row.dot(subspace.row(row));
        subspace.row(other) -= (pivot_row * projection);
        DPP_ASSERT(std::abs(subspace.row(row).dot(subspace.row(other))) <
                   1e-15);
      }
    }

#ifdef DPP_TRACE_SAMPLE
    Eigen::ColPivHouseholderQR<subspace_t> solver;
    solver.compute(subspace_);
    std::cout << "subspace has rank " << solver.rank() << "\n";
    std::cout << "Our subspace is orthogonal: " << is_orthogonal() << "\n";
#endif
  }

  bool is_orthogonal() const {
    subspace_t &subspace_ = this->derived().subspace();
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
  tracer<Fp> *tracer_ = 0;

protected:
  void trace(const Fp * const data, i64 size, TraceType ttype) {
    if (tracer_) {
      tracer_->trace(data, size, ttype);
    }
  }

public:
  void register_tracer(tracer<Fp> *tracer) { tracer_ = tracer; }
};

inline std::vector<i64> greedy_basis_indices(i64 k, i64 n) {
  std::vector<i64> indices(k);
  for (i64 i = 0; i < k; ++i) {
    indices[i] = n - i;
  }
  return indices;
}

}