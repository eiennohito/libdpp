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
    }
  }

  matrix_t &probs() { return probs_; }
};

TEST_F(KernelTest, Load) {
  using namespace dpp;
  float data[] = { 1, 2, 6, 2, 4, 1, 6, 1, 9 };

  auto k = l_kernel<float>::from_array(data, 3);
  auto sampler = k->sampler();
  auto items = sampler.sample();
  delete k;
}

TEST_F(KernelTest, RandomKdpp) {

  Eigen::MatrixXd mat = Eigen::MatrixXd::Random(10, 10);
  Eigen::MatrixXd mat2 = mat.adjoint() * mat;

  std::unique_ptr<dpp::l_kernel<double> > kern(
      dpp::l_kernel<double>::from_array(mat2.data(), 10));

  auto ksamp = kern->sampler(5);

  auto tracer = std::make_shared<storing_tracer<double> >();
  ksamp.register_tracer(tracer.get());

  auto items = ksamp.sample();

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
