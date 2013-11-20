//
//  kernels.cpp
//  libdpp
//
//  Created by Arseny Tolmachev on 2013-11-11.
//
//

#include "gtest/gtest.h"

#include "kernels.h"
#include <Dense>

#include <algorithm>
#include <memory>

template <typename T> std::unique_ptr<T> wrap_ptr(T *ptr) {
  return std::unique_ptr<T>(ptr);
}

class KernelTest : public ::testing::Test {};

template <typename Fp> class storing_tracer : public dpp::tracer<Fp> {
  typedef typename Eigen::Matrix<Fp, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
  typedef typename Eigen::Matrix<Fp, 1, Eigen::Dynamic> vector_t;

  matrix_t vectors_;
  matrix_t probs_;

public:
  void trace(Fp *data, i64 size, dpp::TraceType type) override {
    typedef Eigen::Map<vector_t> vector_map;

    if (type == dpp::TraceType::ProbabilityDistribution) {
      auto rows = probs_.rows();
      probs_.conservativeResize(rows + 1, size);
      probs_.row(rows) = vector_map(data, size);
      probs_.row(rows).normalize();
    }
  }

  matrix_t &probs() { return probs_; }
};

TEST_F(KernelTest, Load) {
  using namespace dpp;
  float data[] = { 1, 2, 6, 2, 4, 1, 6, 1, 9 };

  auto k = l_kernel<float>::from_array(data, 3);
  auto sampler = k->sampler();
  auto items = sampler->sample();
  delete k;
}

TEST_F(KernelTest, RandomKdpp) {

  Eigen::MatrixXd mat = Eigen::MatrixXd::Random(10, 10);
  Eigen::MatrixXd mat2 = mat.adjoint() * mat;

  std::unique_ptr<dpp::l_kernel<double> > kern(
      dpp::l_kernel<double>::from_array(mat2.data(), 10));

  auto ksamp = wrap_ptr(kern->sampler(5));

  auto tracer = std::make_shared<storing_tracer<double> >();
  ksamp->register_tracer(tracer.get());

  auto items = ksamp->sample();

  std::cout << "Probabilities are:\n" << tracer->probs() << "\n";

  std::cout << "Selection is [";
  for (auto i : items) {
    std::cout << i << ",";
  }
  std::cout << "]\n";

  EXPECT_EQ(5, items.size());

  std::sort(std::begin(items), std::end(items));

  std::vector<i64> unique;
  std::unique_copy(std::begin(items), std::end(items),
                   std::back_inserter(unique));
  EXPECT_EQ(5, unique.size());
}

TEST_F(KernelTest, DualKernel) {
  const i64 dim = 6;
  double data[][dim] = { { 1, 0, 0, 0, 0.1, 0 },
                         { 0, 1, 1, 0, 0, 0 },
                         { 0, 0, 1, 0, 0, 0.2 },
                         { 1, 1, 0, 0.1, 0, 0 },
                         { 1, 1, 1, 0, 0, 0.1 },
                         { 1, 0, 0.1, 0, 1, 0 },
                         { 0.1, 1, 1, 0, 0.1, 0 },
                         { 0.1, 1, 0.1, 0, 0, 0 },
                         { 1, 1, 1, 1, 1, 1 },
                         { 1, 0, 0, 0, 1, 1 },
                         { 0, 0.1, 0, 1, 1, 1 } };

  i64 idxs[] = { 0, 1, 2, 3, 4, 5 };

  dpp::c_kernel_builder<double> bldr(dim, dim);
  i64 row_cnt = sizeof(data) / (sizeof(double) * dim);
  bldr.hint_size(row_cnt);
  for (i64 row = 0; row < row_cnt; ++row) {
    bldr.append(data[row], idxs, dim);
  }

  auto kernel = wrap_ptr(bldr.build_kernel());
  auto sampler = wrap_ptr(kernel->sampler(3));

  auto res = sampler->sample();

  std::cout << "Selected: ";
  for (auto i : res) {
    std::cout << i << "\t";
  }
  std::cout << "\n";
}
