#include "libdpp_c.h"
#include "kernels.h"

dpp_kernel_builder dpp_create_kernel_builder(i64 dim_from, i64 dim_to) {
  return new dpp::c_kernel_builder<double>(dim_from, dim_to);
}

void dpp_free_kernel_builder(dpp_kernel_builder builder) {
  auto ptr = reinterpret_cast<dpp::c_kernel_builder<double> *>(builder);
  delete ptr;
}

void dpp_builder_hint_size(dpp_kernel_builder builder, i64 size) {
  auto ptr = reinterpret_cast<dpp::c_kernel_builder<double> *>(builder);
  ptr->hint_size(size);
}

void dpp_builder_append(dpp_kernel_builder builder, double *data, i64 *indices,
                        i64 size) {
  auto ptr = reinterpret_cast<dpp::c_kernel_builder<double> *>(builder);
  ptr->append(data, indices, size);
}

dpp_c_kernel dpp_build_c_kernel(dpp_kernel_builder builder) {
  auto ptr = reinterpret_cast<dpp::c_kernel_builder<double> *>(builder);
  return ptr->build_kernel();
}

dpp_c_sampler dpp_make_c_sampler(dpp_c_kernel kernel) {
  auto ptr = reinterpret_cast<dpp::c_kernel<double> *>(kernel);
  return ptr->sampler();
}

dpp_c_sampler dpp_make_c_sampler_k(dpp_c_kernel kernel, i64 k) {
  auto ptr = reinterpret_cast<dpp::c_kernel<double> *>(kernel);
  return ptr->sampler(k);
}

void dpp_free_c_kernel(dpp_c_kernel kernel) {
  auto ptr = reinterpret_cast<dpp::c_kernel<double> *>(kernel);
  delete ptr;
}

void dpp_free_c_sampler(dpp_c_sampler sampler) {
  auto ptr = reinterpret_cast<dpp::dual_sampling_subspace<double> *>(sampler);
  delete ptr;
}

void dpp_perform_c_sampling(dpp_c_sampler sampler, dpp_sample_result res) {
  auto ptr = reinterpret_cast<dpp::dual_sampling_subspace<double> *>(sampler);
  auto vec = reinterpret_cast<std::vector<i64> *>(res);
  ptr->sample(*vec);
}

dpp_sample_result dpp_make_sample_result() { return new std::vector<i64>(); }

void dpp_free_sample_result(dpp_sample_result result) {
  auto ptr = reinterpret_cast<std::vector<i64> *>(result);
  delete ptr;
}

i64 dpp_sample_size(dpp_sample_result result) {
  auto ptr = reinterpret_cast<std::vector<i64> *>(result);
  return ptr->size();
}

i64 *dpp_sample_data(dpp_sample_result result) {
  auto ptr = reinterpret_cast<std::vector<i64> *>(result);
  return ptr->data();
}