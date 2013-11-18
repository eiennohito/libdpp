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

template <typename Derived> class base_object {
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

public:
  mutable std::mt19937 rng_{ std::random_device()() };

  std::vector<i64> random_subspace_indices() const {
    std::vector<i64> vec;
    vec.reserve(static_cast<i64>(
        sqrt(this->derived().kernel().rows()))); // reserve some memory
    auto &eigen_values = eigenvalues().array();

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
    std::cout << "Eigenvalues are: " << ev << "\n";
    std::cout << "Computed elementary polynomials:\n" << pols << "\n";
    
    kernel_t probs(nrow, k);
    for (i64 l = 0; l < k; ++l) {
      for (i64 n = 0; n < nrow; ++n) {
        probs(n, l) = ev(n) * pols(n, l) / pols(n + 1, l + 1);
      }
    }
    
    std::cout << "Probabilities to select are:\n" << probs << "\n";
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
    auto &ep = *polynomials_; // 1-based matrix
    auto N = ev.size();
    for (i64 n = N - 1; n >= 0;
         --n) { // because of zero-based indices n is less by 1
      if (l == 0)
        break;
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

  void decompose() { eigen_.compute(this->derived().kernel()); }

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
};

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
};

template <typename Fp>
l_kernel<Fp> *l_kernel<Fp>::from_array(Fp *data, i64 size) {
  auto impl = make_unique<typename l_kernel<Fp>::impl_t>();
  impl->init_from_kernel(data, size, size);
  impl->decompose();
  return new l_kernel<Fp>(std::move(impl));
}

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
  tracer<Fp> *tracer_;

protected:
  void trace(Fp *data, i64 size, TraceType ttype) {
    if (tracer_) {
      tracer_->trace(data, size, ttype);
    }
  }

public:
  void register_tracer(tracer<Fp> *tracer) { tracer_ = tracer; }
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
      : kernel_{ kernel }, vec_indices_{ std::move(idxs) } {
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

  void sample(std::vector<i64> &buffer) {
    i64 height = subspace_.rows();
    buffer.reserve(height);

    for (i64 i = 0; i < height; ++i) {
#ifdef DPP_TRACE_SAMPLE
      std::cout << "Initial subspace\n" << subspace_ << "\n";
#endif

      cached_ = subspace_.colwise().squaredNorm(); // calculate probabilities

#ifdef DPP_TRACE_SAMPLE
      std::cout << "weights for the selection are:\n" << cached_ << "\n";
#endif

      this->trace(cached_.data(), cached_.size(),
                  TraceType::ProbabilityDistribution);

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
  return new sampling_subspace_impl<Fp>{ this,
                                         this->random_subspace_indices() };
}

template <typename Fp>
sampling_subspace_impl<Fp> *l_kernel_impl<Fp>::sampler(i64 k) {
  return new sampling_subspace_impl<Fp>{ this,
                                         this->k_random_subspace_indices(k) };
}

template <typename Fp>
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

template <typename Fp>
class c_kernel_impl : public base_kernel<c_kernel_impl<Fp>, Fp> {
public:
  typedef typename Eigen::Matrix<Fp, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor> matrix_t;
  typedef typename base_kernel<c_kernel_impl<Fp>, Fp>::kernel_t kernel_t;

private:
  // dual DPP-kernel, has dimensions of D \times D
  kernel_t kernel_;

  // dense matrix of items, each row contains
  // vector similarity features (norm == 1) multiplied
  // by scalar quality features, so the norm == quality.
  // Dimensions of matrix are N \times D
  matrix_t matrix_;

public:
  c_kernel_impl(matrix_t &&matrix) : matrix_{ std::move(matrix) } {
    kernel_ = matrix_.adjoint() * matrix_;
  }

  kernel_t &kernel() { return kernel_; }
  const kernel_t &kernel() const { return kernel_; }

  matrix_t &matrix() { return matrix_; }
  const matrix_t &matrix() const { return matrix_; }
};

template <typename Fp>
class c_sampler_impl : public base_sampler_impl<c_sampler_impl<Fp>, Fp> {

  typedef typename base_sampler_impl<c_sampler_impl<Fp>, Fp>::subspace_t
  subspace_t;
  typedef Eigen::Matrix<Fp, 1, Eigen::Dynamic> vector_t;

  // dual kernel operates in terms of energy products (a' * C * b) instead of
  // dot products
  // here C is our dual DPP-kernel, a and b are D-dimensional vectors
  template <typename Vec1, typename Vec2>
  Fp energy_product(Vec1 &&v1, Vec2 &&v2) const {
    auto &&product = (v1 * kernel_->kernel());
    DPP_ASSERT(product.cols() == v2.cols() && product.rows() == 1);
    return product.row(0).dot(v2);
  }

  std::vector<i64> items_;

  // has the dimensions of k \times D, k number of items
  subspace_t subspace_;

  vector_t probs_;
  vector_t temp_;

  c_kernel_impl<Fp> *kernel_;

  void reset() {
    // refill subspace
    i64 size = items_.size();
    subspace_.resize(size, kernel_->cols());
    for (i64 i = 0; i < size; ++i) {
      subspace_.row(i) = kernel_->eigenvector(i);
      Fp len = std::sqrt(energy_product(subspace_.row(i), subspace_.row(i)));
      subspace_.row(i) /= len;
    }
  }

public:
  c_sampler_impl(c_kernel_impl<Fp> *kernel, std::vector<i64> &&items)
      : kernel_{ kernel }, items_{ std::move(items) } {}

  void sample(std::vector<i64> &res) {
    reset();
    
    res.clear();

    i64 times = subspace_.rows();
    res.reserve(times);

    auto &rng = kernel_->rng_;

    for (i64 i = 0; i < times; ++i) {
      probs_ =
          (subspace_ * kernel_->matrix().adjoint()).colwise().squaredNorm();
      Fp sum = probs_.sum();
      std::uniform_real_distribution<Fp> distr{ 0, sum };
      Fp randval = distr(rng);
      Fp cur = 0;

      i64 selected = 0;
      for (; selected < probs_.size(); ++selected) {
        cur += probs_[selected];
        if (randval < cur) {
          break;
        }
      }

      res.push_back(selected);
      remove_item_from_subspace(selected);
      orthogonalize_subspace();
    }
  }

  void remove_item_from_subspace(i64 selected) {
    auto &&item = kernel_->matrix().row(selected);

    i64 height = subspace_.rows();
    i64 pivot;
    Fp pivot_prod;
    for (pivot = 0; pivot < height; ++pivot) {
      pivot_prod = item.dot(subspace_.row(pivot));
      if (std::abs(pivot_prod) > 1e-6) {
        break;
      }
    }

    auto &&pivot_row = subspace_.row(pivot);

    for (i64 i = 0; i < height; ++i) {
      if (i != pivot) {
        auto &&row = subspace_.row(i);
        Fp sim = item.dot(row);
        row -= (pivot_row * sim / pivot_prod);
      }
    }

    pivot_row.setZero();
  }

  void orthogonalize_subspace() {
    auto height = subspace_.rows();
    for (i64 row = 0; row < height; ++row) {
      auto &&pivot_row = subspace_.row(row);

      auto norm = std::sqrt(energy_product(pivot_row, pivot_row));
      if (norm > 1e-10) {
        pivot_row /= norm;
      } else {
        pivot_row.setZero();
        continue;
      }
      // pivot_row.normalize();

      // DPP_ASSERT(std::abs(subspace_.row(row).norm() - 1) < 1e-15);
      for (i64 other = row + 1; other < height; ++other) {
        auto &&other_row = subspace_.row(other);
        auto projection = energy_product(other_row, subspace_.row(row));
        subspace_.row(other) -= (pivot_row * projection);
        DPP_ASSERT(std::abs(energy_product(subspace_.row(row),
                                           subspace_.row(other))) < 1e-15);
      }
    }
  }
};

template <typename Fp> class c_kernel_builder_impl {
  typedef typename c_kernel_impl<Fp>::matrix_t matrix_t;
  typedef typename c_kernel_impl<Fp>::kernel_t projection_t;

  matrix_t matrix_;
  projection_t projection_;

  i64 from_ = 0, to_ = 0, row_ = 0;

  void check_sizes() {
    if (matrix_.rows() <= row_) {
      matrix_.conservativeResize(row_ + 1, to_);
    }
  }

  void init_random_projection() {
    std::mt19937 rng;
    std::normal_distribution<Fp> distr{ 0, 1.0 / to_ };
    projection_.setZero(from_, to_);
    for (i64 i = 0; i < from_; ++i) {
      for (i64 j = 0; j < to_; ++j) {
        projection_(i, j) = distr(rng);
      }
    }
  }

public:
  c_kernel_builder_impl(i64 dim_from, i64 dim_to)
      : from_{ dim_from }, to_{ dim_to } {
    if (from_ == to_) {
      projection_ = matrix_t::Identity(from_, from_);
    } else {
      DPP_ASSERT(from_ > to_);
      init_random_projection();
    }
  }

  void hint_size(i64 data_size) { matrix_.conservativeResize(data_size, to_); }

  void append(Fp *coeff, i64 *indices, i64 size) {
    check_sizes();
    auto &&row = matrix_.row(row_);
    ++row_;
    row.setZero();
    for (i64 i = 0; i < size; ++i) {
      i64 idx = indices[i];
      DPP_ASSERT(idx >= 0);
      DPP_ASSERT(idx < from_);
      Fp val = coeff[i];
      row += val * projection_.row(idx);
    }
  }

  c_kernel_impl<Fp> *build() {
    matrix_.conservativeResize(row_, to_);
    return new c_kernel_impl<Fp>(std::move(matrix_));
  }
};

template <typename Fp>
c_kernel_builder<Fp>::c_kernel_builder(i64 from, i64 to)
    : impl_{ new c_kernel_builder_impl<Fp>(from, to) } {}

template <typename Fp>
void c_kernel_builder<Fp>::append(Fp *data, i64 *idxes, i64 size) {
  impl_->append(data, idxes, size);
}

template <typename Fp> c_kernel<Fp> *c_kernel_builder<Fp>::build_kernel() {
  c_kernel_impl<Fp> *impl = impl_->build();
  impl->decompose();
  return new c_kernel<Fp>(impl);
}

template <typename Fp> void c_kernel_builder<Fp>::hint_size(i64 size) {
  impl_->hint_size(size);
}

template <typename Fp> c_kernel_builder<Fp>::~c_kernel_builder() {}

template <typename Fp> dual_sampling_subspace<Fp> *c_kernel<Fp>::sampler() {
  auto &&items = impl_->random_subspace_indices();
  auto impl = new c_sampler_impl<Fp>(impl_.get(), std::move(items));
  return new dual_sampling_subspace<Fp>(make_unique(impl));
}

template <typename Fp>
dual_sampling_subspace<Fp> *c_kernel<Fp>::sampler(i64 k) {
  auto &&items = impl_->k_random_subspace_indices(k);
  auto impl = new c_sampler_impl<Fp>(impl_.get(), std::move(items));
  return new dual_sampling_subspace<Fp>(make_unique(impl));
}

template <typename Fp> c_kernel<Fp>::~c_kernel<Fp>() {}

template <typename Fp>
void dual_sampling_subspace<Fp>::sample(std::vector<i64> &res) {
  impl_->sample(res);
}

template <typename Fp> std::vector<i64> dual_sampling_subspace<Fp>::sample() {
  std::vector<i64> res;
  sample(res);
  return std::move(res);
}

template <typename Fp> dual_sampling_subspace<Fp>::~dual_sampling_subspace() {}

template class l_kernel<float>;
template class l_kernel<double>;

template class sampling_subspace<float>;
template class sampling_subspace<double>;

template class c_kernel_builder<float>;
template class c_kernel_builder<double>;

template class c_kernel<float>;
template class c_kernel<double>;

template class dual_sampling_subspace<float>;
template class dual_sampling_subspace<double>;
}