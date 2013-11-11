#ifndef DPP_KERNEL_H
#define DPP_KERNEL_H 1


#include <memory>

#include <cstdint>

using i64 = std::int64_t;

//#include <boost/preprocessor.hpp>

namespace dpp {

  template <typename Fp>
  class l_kernel_impl;

  template <typename Fp>
  class l_kernel {
    typedef typename dpp::l_kernel_impl<Fp> impl_t;
    typedef typename std::unique_ptr<impl_t> impl_ptr;

    impl_ptr impl_;

    l_kernel(impl_ptr&& impl): impl_{std::move(impl)} {}

  public:
    static l_kernel* from_array(Fp* data, int rows, int cols);
  };

  extern template class l_kernel<float>;
  extern template class l_kernel<double>;
}

#endif