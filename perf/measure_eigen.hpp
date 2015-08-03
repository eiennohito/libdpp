//
// Created by Arseny Tolmachev on 2015/08/03.
//

#ifndef LIBDPP_MEASURE_EIGEN_HPP
#define LIBDPP_MEASURE_EIGEN_HPP

#include "perf_common.hpp"

measure_stats measure_mult_eig(int rows, int cols, int times);
measure_stats measure_svd_jacobi_thinv(int rows, int cols, int times);
measure_stats measure_svd_d_thinv(int rows, int cols, int times);

#endif //LIBDPP_MEASURE_EIGEN_HPP
