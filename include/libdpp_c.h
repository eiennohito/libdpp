#ifndef __LIBDPP__LIBDPP_C__
#define __LIBDPP__LIBDPP_C__

#ifdef __cplusplus
extern "C" {
#endif

typedef long long i64;

typedef void *dpp_c_kernel;
//typedef void *dpp_c_selector;

dpp_c_kernel dpp_create_c_kernel(double *data, i64 ndim, i64 size);
void dpp_delete_c_kernel(dpp_c_kernel kernel);

void dpp_select_from_c(dpp_c_kernel kernel, i64 nitems, i64* buffer);

#ifdef __cplusplus
}
#endif

#endif