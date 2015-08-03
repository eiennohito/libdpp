//
// Created by Arseny Tolmachev on 2015/08/03.
//

#include "measure_eigen.hpp"
#include "perf_common.hpp"

#include <Eigen/Dense>



measure_stats measure_mult_eig(int rows, int cols, int times) {
  Eigen::MatrixXd mat;
  mat.setRandom(rows, cols);

  auto mfunc = [&]() {
    Eigen::MatrixXd m2 = mat.adjoint() * mat;
    assert(m2.rows() == cols);
    assert(m2.cols() == cols);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(m2, Eigen::ComputeEigenvectors);
    return eigs.info();
  };

  auto perf = make_measurer(mfunc);
  return perf.run(times);
}

measure_stats measure_svd_jacobi_thinv(int rows, int cols, int times) {
  Eigen::MatrixXd mat;
  mat.setRandom(rows, cols);

  auto mfunc = [&]() {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinV);
    return svd.rank();
  };

  auto perf = make_measurer(mfunc);
  return perf.run(times);
}

measure_stats measure_svd_d_thinv(int rows, int cols, int times) {
  Eigen::MatrixXd mat;
  mat.setRandom(rows, cols);

  auto mfunc = [&]() {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinV);
    return svd.rank();
  };

  auto perf = make_measurer(mfunc);
  return perf.run(times);
}
