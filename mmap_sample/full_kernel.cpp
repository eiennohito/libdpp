//
// Created by Arseny Tolmachev on 2015/04/10.
//

#include <kernels.h>
#include <iostream>

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <boost/lexical_cast.hpp>
#include <memory>

#include "Eigen/Core"

namespace ip = boost::interprocess;

template <typename R>
std::unique_ptr<R> wrap(R* ptr) {
  return std::unique_ptr<R>(ptr);
}

/**
 * An entry point to the sampler
 * 4 arguments:
 * 1) ndim of each feature vector
 * 2) number of feature vectors
 * 3) path to file that holds vector data as native endian doubles
 * 4) path to file that is going to be used as space for output, at least of the length of feature vectors + 1,
 *    each one is of 64bit.
 *    The first one is going to be the size of result data, following are
 *    the selection itself.
 * 5) number of samples (20 default)
 */
int main(int argc, char **argv) {

  size_t ndim = boost::lexical_cast<size_t>(argv[1]);
  size_t nvecs = boost::lexical_cast<size_t>(argv[2]);

  char *data_path = argv[3];
  ip::file_mapping data(data_path, ip::read_only);
  ip::mapped_region data_reg(data, ip::read_only);

  char *output_path = argv[4];
  ip::file_mapping output(output_path, ip::read_write);
  ip::mapped_region out_reg(output, ip::read_write);

  i64 nsamples = 20;

  if (argc == 6) {
    nsamples = boost::lexical_cast<decltype(nsamples)>(argv[5]);
  }

  if (argc != 6) {
    return 1;
  }

  double *data_ptr = reinterpret_cast<double *>(data_reg.get_address());

  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;
  matrix_t mat;

  Eigen::Map<matrix_t> raw(data_ptr, ndim, nvecs);
  mat = raw.adjoint() * raw;

  auto kernel = wrap(dpp::l_kernel<double>::from_array(mat.data(), nvecs));
  auto sampler = wrap(kernel->sampler());

  std::vector<i64> out_vec;

  sampler->greedy_prob_selection(out_vec, nsamples);

  i64 *ptr = reinterpret_cast<i64 *>(out_reg.get_address());
  std::copy(out_vec.begin(), out_vec.end(), ptr + 1);
  *ptr = (i64) out_vec.size();

  out_reg.flush();

  return 0;
};
