set( sources
  libdpp_c.cpp
  common.hpp
  c_kernel.cpp
  l_kernel.cpp
  results.cpp
)

set ( our_includes
  ../include/kernels.h
  ../include/libdpp_c.h
  common.hpp
  selector_base.hpp
  ../include/results.hpp
)

include_directories(${libdpp_INCLUDES})

add_library(dpp ${sources} ${our_includes} ../include/fwddefs.hpp)