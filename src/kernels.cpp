#include "kernels.h"

// use Eigen's dense matrices
#include "Dense"
//#include "Map.h"

// use Eigen's eigendecomposition
#include <Eigenvalues>

#include <random>
#include <vector>

namespace dpp {

template <typename R, typename... T>
std::unique_ptr<R> make_unique(T&&... vals) {
  return std::unique_ptr<R>(new R(std::forward<T>(vals)...));
}
  
  template <typename Fp> class sampling_subspace_impl;

template <typename Fp>
struct eigen_typedefs {
  typedef Eigen::Matrix<Fp, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      matrix_rowmajor;
  typedef Eigen::Matrix<Fp, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
      matrix_colmajor;
  
  typedef Eigen::Matrix<Fp, Eigen::Dynamic, 1> vector;
};

template <typename Fp>
class l_kernel_impl {

  typedef typename eigen_typedefs<Fp>::matrix_colmajor kernel_t;
  std::unique_ptr<kernel_t> kernel_;
  Eigen::SelfAdjointEigenSolver<kernel_t> eigen_;
  
  std::mt19937 rng_{std::random_device()()};

  
 public:
  void init_from_kernel(Fp* data, int rows, int cols) {
    typedef typename eigen_typedefs<Fp>::matrix_rowmajor outer_t;
    Eigen::Map<outer_t> outer(data, rows, cols);
    kernel_ = make_unique<kernel_t>(rows, cols);
    *kernel_ = outer;
  }

  void decompose() {
    eigen_.compute(*kernel_);
  }
  
  std::vector<i64> random_subspace_indices() const {
    std::vector<i64> vec;
    vec.reserve(static_cast<i64>(sqrt(kernel_->rows()))); //reserve some memory
    auto &eigen_values = eigen_.eigenvalues();
    auto probs = eigen_values / (eigen_values + 1);
    
    std::uniform_real_distribution<Fp> distr{0, 1};
    for (i64 i = 0; i < eigen_values.cols(); ++i) {
      auto rn = distr(rng_);
      if (rn < probs[i]) {
        vec.push_back(i);
      }
    }
    
    return std::move(vec);
  }
  
  auto eigenvector(i64 pos) -> decltype(eigen_.eigenvectors().row(pos)) const {
    return eigen_.eigenvectors().row(pos);
  }
  
  
  sampling_subspace_impl<Fp> sampler() const {
    return sampling_subspace_impl<Fp> {this, random_subspace_indices()};
  }
  
};

template <typename Fp>
l_kernel<Fp>* l_kernel<Fp>::from_array(Fp* data, int rows, int cols) {
  auto impl = make_unique<typename l_kernel<Fp>::impl_t>();
  impl->init_from_kernel(data, rows, cols);
  return new l_kernel<Fp>(std::move(impl));
}

template class l_kernel<float>;
template class l_kernel<double>;
  
  template <typename Fp>
  class sampling_subspace_impl {
    typedef typename eigen_typedefs<Fp>::matrix_rowmajor subspace_t;
    
    subspace_t subspace_;
    const l_kernel_impl<Fp> *kernel_;
    const std::vector<i64> vec_indices_;
    
    sampling_subspace_impl(l_kernel_impl<Fp>* kernel, std::vector<i64>&& idxs):
    kernel_{kernel}, vec_indices_{std::move(idxs)}, subspace_{idxs.size(), kernel->cols()} {
      reset();
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
        auto& pivot_row = subspace_.row(row);
        pivot_row.normalize();
        for (i64 other = row + 1; other < row; ++other) {
          auto &other_row = subspace_.row(other);
          auto projection = other_row.dot(pivot_row);
          other_row -= pivot_row * projection;
        }
      }
    }
    
    
    typedef typename eigen_typedefs<Fp>::vector vector_t;
    vector_t cached_;
    
    /***
     Compute a subspace that is orthogonal to the i-th basis vector.
     The i-th basis vector has all zeros except only one in the i-th component.
     
     This function computes a projection of e_i to the subspace spanned by the vectors,
     then does one step of the Gram-Shmidt process modifying the subspace to be
     orthogonal to e_i.
    */
    void orthogonal_subspace(i64 comp) {
      i64 height = subspace_.rows();
      //compute a projection of a basis vector e_{comp} to the subspace
      cached_ = vector_t::Zero(subspace_.cols());
      for (i64 i = 0; i < height; ++i) {
        cached_ += subspace_.row(i) * subspace_(i, comp);
      }
      
      cached_.normalize();
      for (i64 i = 0; i < height; ++i) {
        auto &row = subspace_.row(i);
        auto proj = row.dot(cached_);
        row -= cached_ * proj;
      }
    }
    
    
    void sample(std::vector<i64> &buffer) {
      i64 height = subspace_.rows();
      buffer.reserve(height);
      
      for (i64 i = 0; i < height; ++i) {
        cached_ = subspace_.colwise().squaredNorm(); //calculate probabilities
        auto len = cached_.sum();
        std::uniform_real_distribution<Fp> distr{0, len};
        auto prob = distr(kernel_->rng_);
        i64 selected = 0;
        Fp val = 0;
        for (;selected < cached_.cols(); ++selected) {
          val += cached_[selected];
          if (val > prob) break;
        }
        
        buffer.push_back(selected);
        orthogonal_subspace(selected);
        gram_shmidt_orhonormailze();
      }
    }

  };
  
  
}