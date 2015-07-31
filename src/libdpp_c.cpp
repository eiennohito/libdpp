#include "libdpp_c.h"
#include "kernels.h"

dpp_c_kernel dpp_create_c_kernel(double *data, i64 ndim, i64 size) {
  auto kern = dpp::make_c_kernel(data, ndim, size);
  return kern.release();
}

void dpp_delete_c_kernel(dpp_c_kernel kernel) {
  delete reinterpret_cast<dpp::c_kernel<double>*>(kernel);
}

void dpp_select_from_c(dpp_c_kernel kernel, i64 nitems, i64 *buffer) {
  auto kern = reinterpret_cast<dpp::c_kernel<double>*>(kernel);
  auto sel = kern->selector();
  dpp::memory_area_result_holder hldr(buffer, nitems);
  sel->greedy_max_subset(nitems, hldr);
}
