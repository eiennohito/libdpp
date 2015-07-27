#include "common.hpp"

namespace dpp {



template <typename Fp>
class c_kernel_impl : public base_kernel<c_kernel_impl<Fp>, Fp> {
public:
  typedef typename Eigen::Matrix<Fp, Eigen::Dynamic, Eigen::Dynamic,
      Eigen::RowMajor> matrix_t;
  typedef typename eigen_typedefs<Fp>::matrix_colmajor kernel_t;

private:
  // dual DPP-kernel, has dimensions of D \times D
  kernel_t kernel_;

  // dense matrix of items, each row contains
  // vector similarity features (norm == 1) multiplied
  // by scalar quality features, so the norm == quality.
  // Dimensions of matrix are N \times D
  matrix_t matrix_;

public:
  c_kernel_impl(matrix_t &&matrix) : matrix_{std::move(matrix)} {
    kernel_ = matrix_.adjoint() * matrix_;
  }

  kernel_t &kernel() { return kernel_; }
  const kernel_t &kernel() const { return kernel_; }

  matrix_t &matrix() { return matrix_; }
  const matrix_t &matrix() const { return matrix_; }

  Fp selection_log_probability(const std::vector<i64> &indices) const {

    //1. create a reduction kernel
    kernel_t reduced(indices.size(), indices.size());
    reduced.setZero();

    auto sz = indices.size();
    auto ndim = this->matrix().cols();

    kernel_t selected_items;
    selected_items.resize(sz, ndim);

    for (i64 i = 0; i < sz; ++i) {
      selected_items.row(i) = this->matrix().row(indices[i]);
    }

    kernel_t mods = this->eigenvectors();

    for (i64 i = 0; i < mods.rows(); ++i) {
      mods.col(i).array() /= std::sqrt(1 + this->eigenvalues()(i));
    }

    /*

    typename eigen_typedefs<Fp>::vector vec(sz);

    for (i64 n = 0; n < ndim; ++n) {

      auto &&evec = this->eigenvector(n);
      vec = selected_items * evec.adjoint();
      auto mplier = 1 / (this->eigenvalues()(n) + 1);

      for (i64 i = 0; i < sz; ++i) {
        auto step_mplier = mplier * vec(i);
        reduced(i, i) += step_mplier * vec(i);
        for (i64 j = i + 1; j < sz; ++j) {
          auto val = step_mplier * vec(j);
          reduced(i, j) += val;
          reduced(j, i) += val;
        }
      }
    } */

    selected_items *= mods;

    reduced = selected_items * selected_items.adjoint();

    Eigen::LDLT<kernel_t> distr(reduced);

    //2. return result
    return distr.vectorD().array().log().sum();
  }

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
  Fp energy_product(Vec1 &&v1, Vec2 &&v2) {
    mult_temp_.noalias() = (v1 * kernel_->kernel());
    // DPP_ASSERT(product.cols() == v2.cols() && product.rows() == 1);
    return mult_temp_.dot(v2);
  }

  std::vector<i64> items_;

  // has the dimensions of k \times D, k number of items
  subspace_t subspace_;

  vector_t probs_;
  vector_t temp_, mult_temp_;

  c_kernel_impl<Fp> *kernel_;

  void reset() {
    // refill subspace
    i64 size = items_.size();
    subspace_.resize(size, kernel_->cols());
    for (i64 i = 0; i < size; ++i) {
      subspace_.row(i) = kernel_->eigenvector(items_[i]);
      Fp len = std::sqrt(energy_product(subspace_.row(i), subspace_.row(i)));
      subspace_.row(i) /= len;
    }
  }

public:
  c_sampler_impl(c_kernel_impl<Fp> *kernel, std::vector<i64> &&items)
      : items_{std::move(items)}, kernel_{kernel} {}

  void sample(std::vector<i64> &res, bool greedy = false) {
    reset();

    res.clear();

    i64 times = subspace_.rows();
    res.reserve(times);

    auto &rng = kernel_->rng_;

    for (i64 i = 0; i < times; ++i) {
      probs_ =
          (subspace_ * kernel_->matrix().adjoint()).colwise().squaredNorm();

      this->trace(probs_.data(), probs_.size(),
                  TraceType::ProbabilityDistribution);

      i64 selected = 0;
      if (greedy) {
        Fp maxsel = 0;
        i64 cand = 0;
        for (; cand < probs_.size(); ++cand) {
          Fp val = probs_[cand];
          if (val > maxsel) {
            maxsel = val;
            selected = cand;
          }
        }
      } else {
        Fp sum = probs_.sum();
        std::uniform_real_distribution<Fp> distr{0, sum};
        Fp randval = distr(rng);
        Fp cur = 0;

        for (; selected < probs_.size(); ++selected) {
          cur += probs_[selected];
          if (randval < cur) {
            break;
          }
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
    i64 pivot, sel_pivot;
    Fp pivot_prod = 0;
    for (pivot = 0; pivot < height; ++pivot) {
      Fp test = item.dot(subspace_.row(pivot));
      if (std::abs(test) > std::abs(pivot_prod)) {
        pivot_prod = test;
        sel_pivot = pivot;
      }
    }

    pivot = sel_pivot;

    auto &&pivot_row = subspace_.row(pivot);

    for (i64 i = 0; i < height; ++i) {
      if (i != pivot) {
        auto &&row = subspace_.row(i);
        Fp sim = item.dot(row);
        row -= (pivot_row * sim / pivot_prod);

        DPP_ASSERT(std::abs(item.dot(row)) < 1e-10);
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
        auto projection = energy_product(other_row, pivot_row);
        subspace_.row(other) -= (pivot_row * projection);
        DPP_ASSERT(std::abs(energy_product(subspace_.row(other), pivot_row)) <
                   1e-10);
      }
    }
  }
};

template <typename Fp>
class c_kernel_builder_impl {
  typedef typename c_kernel_impl<Fp>::matrix_t matrix_t;
  typedef typename c_kernel_impl<Fp>::kernel_t projection_t;

  matrix_t matrix_;
  projection_t projection_;

  i64 from_ = 0, to_ = 0, row_ = 0;

  void check_sizes() {
    if (matrix_.rows() <= row_) {
      i64 sz = std::max(static_cast<i64>(row_ * 1.4), row_ + 5);
      matrix_.conservativeResize(sz, to_);
    }
  }

  void init_random_projection() {
    std::mt19937 rng;
    std::normal_distribution<Fp> distr{0, Fp{1.0} / to_};
    projection_.setZero(from_, to_);
    for (i64 i = 0; i < from_; ++i) {
      for (i64 j = 0; j < to_; ++j) {
        projection_(i, j) = distr(rng);
      }
    }
  }

public:
  c_kernel_builder_impl(i64 dim_from, i64 dim_to)
      : from_{dim_from}, to_{dim_to} {
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
    : impl_{new c_kernel_builder_impl<Fp>(from, to)} {}

template <typename Fp>
void c_kernel_builder<Fp>::append(Fp *data, i64 *idxes, i64 size) {
  impl_->append(data, idxes, size);
}

template <typename Fp>
c_kernel<Fp> *c_kernel_builder<Fp>::build_kernel() {
  c_kernel_impl<Fp> *impl = impl_->build();
  impl->decompose();
  return new c_kernel<Fp>(impl);
}

template <typename Fp>
void c_kernel_builder<Fp>::hint_size(i64 size) {
  impl_->hint_size(size);
}

template <typename Fp>
c_kernel_builder<Fp>::~c_kernel_builder() {}

template <typename Fp>
dual_sampling_subspace<Fp> *c_kernel<Fp>::sampler() {
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

template <typename Fp>
dual_sampling_subspace<Fp> *c_kernel<Fp>::sampler_greedy(i64 k) {
  auto impl = new c_sampler_impl<Fp>(impl_.get(),
                                     greedy_basis_indices(k,
                                                          impl_->kernel().rows()));
  return new dual_sampling_subspace<Fp>(make_unique(impl));
}

template <typename Fp>
c_kernel<Fp>::~c_kernel<Fp>() {}

template <typename Fp>
void dual_sampling_subspace<Fp>::sample(std::vector<i64> &res) {
  impl_->sample(res, false);
}

template <typename Fp>
std::vector<i64> dual_sampling_subspace<Fp>::sample() {
  std::vector<i64> res;
  sample(res);
  return std::move(res);
}

template <typename Fp>
void dual_sampling_subspace<Fp>::greedy(std::vector<i64> &res) {
  impl_->sample(res, true);
}

template <typename Fp>
void dual_sampling_subspace<Fp>::register_tracer(tracer<Fp> *t) {
  impl_->register_tracer(t);
}

template <typename Fp>
dual_sampling_subspace<Fp>::~dual_sampling_subspace() {}

template<typename Fp>
c_kernel<Fp> *c_kernel<Fp>::from_colwize_array(Fp *data, i64 ndim, i64 size) {

  Eigen::Map<typename eigen_typedefs<Fp>::matrix_colmajor, Eigen::Unaligned, Eigen::Stride<1, Eigen::Dynamic>>
      mapped(data, size, ndim, Eigen::Stride<1, Eigen::Dynamic>(1, ndim));

  auto matr = typename c_kernel_impl<Fp>::matrix_t(mapped);
  auto impl_ptr = make_unique<c_kernel_impl<Fp>>(std::move(matr));

  impl_ptr->decompose();

  //auto impl = make_unique<c_kernel_impl>(matrix);
  return new dpp::c_kernel<Fp>(impl_ptr.release());
}

template <typename Fp>
Fp c_kernel<Fp>::selection_log_probability(std::vector<i64> &indices) {
  return this->impl_->selection_log_probability(indices);
}

//explicit instantiations

template class c_kernel_builder<float>;
template class c_kernel_builder<double>;

template class c_kernel<float>;
template class c_kernel<double>;

template class dual_sampling_subspace<float>;
template class dual_sampling_subspace<double>;


}