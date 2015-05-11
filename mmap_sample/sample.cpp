//
// Created by Arseny Tolmachev on 2015/04/10.
//

#include <kernels.h>
#include <iostream>

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <boost/lexical_cast.hpp>
#include <memory>

namespace ip = boost::interprocess;

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


  double *data_ptr = reinterpret_cast<double *>(data_reg.get_address());

  std::unique_ptr<dpp::c_kernel<double>> kernel{
      dpp::c_kernel<double>::from_colwize_array(data_ptr, (i64) ndim, (i64) nvecs)
  };

  std::unique_ptr<dpp::dual_sampling_subspace<double>> sampler{kernel->sampler(20)};

  std::vector<i64> out_vec;

  sampler->greedy(out_vec);

  i64 *ptr = reinterpret_cast<i64 *>(out_reg.get_address());
  std::copy(out_vec.begin(), out_vec.end(), ptr + 1);
  *ptr = (i64) out_vec.size();

  out_reg.flush();

  return 0;
};
