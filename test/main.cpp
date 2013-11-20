#include <iostream>
#include "gtest/gtest.h"

#include <random>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    matrix;

TEST(SomeTest, Test) { EXPECT_EQ(0, 0); }

TEST(SomeTest, Cpp11Feature) {
  auto c = 2;
  auto f = [](int x) { return 3; };
  f(c);
}

TEST(MatrixTest, SimpleTest) {
  matrix mat(10, 10);
  std::mt19937 gen(10);
  std::uniform_real_distribution<float> distr(0, 1);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      mat(i, j) = distr(gen);
    }
  }

  auto mat3 = mat.transpose() * mat;

  Eigen::SelfAdjointEigenSolver<matrix> eig_solver;

  eig_solver.compute(mat3);

  auto eig = eig_solver.eigenvalues();

  //  std::cout << eig << std::endl;
  //  std::cout << mat3 << std::endl;
  //  std::cout << eig_solver.eigenvectors() << std::endl;
}

int main(int argc, char **argv) {
  std::cout << "Running tests from gtest\n";
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}