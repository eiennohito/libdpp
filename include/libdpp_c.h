#ifndef __LIBDPP__LIBDPP_C__
#define __LIBDPP__LIBDPP_C__

#ifdef __cplusplus
extern "C" {
#endif
  
  typedef long long i64;
  //typedef void* dpp_l_kernel;
  //typedef void* dpp_l_sampler;
  typedef void* dpp_kernel_builder;
  typedef void* dpp_c_kernel;
  typedef void* dpp_c_sampler;
  
  typedef void* dpp_sample_result;
  
  dpp_kernel_builder dpp_create_kernel_builder(i64 dim_from, i64 dim_to);
  void dpp_free_kernel_builder(dpp_kernel_builder builder);
  void dpp_builder_hint_size(dpp_kernel_builder builder, i64 size);
  void dpp_builder_append(dpp_kernel_builder builder, double* data, i64* indices, i64 size);
  dpp_c_kernel dpp_build_c_kernel(dpp_kernel_builder builder);
  
  dpp_c_sampler dpp_make_c_sampler(dpp_c_kernel kernel);
  dpp_c_sampler dpp_make_c_sampler_k(dpp_c_kernel kernel, i64 k);
  void dpp_free_c_kernel(dpp_c_kernel kernel);
  
  void dpp_free_c_sampler(dpp_c_sampler sampler);
  void dpp_perform_c_sampling(dpp_c_sampler sampler, dpp_sample_result res);
  
  dpp_sample_result dpp_make_sample_result();
  void dpp_free_sample_result(dpp_sample_result result);
  i64 dpp_sample_size(dpp_sample_result result);
  i64* dpp_sample_data(dpp_sample_result result);
  
#ifdef __cplusplus
}
#endif

#endif