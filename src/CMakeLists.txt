set( sources
  libdpp_c.cpp
  common.hpp
  c_kernel.cpp
  l_kernel.cpp
  results.cpp
  l_selection.cpp
  c_selector.cpp
)

set ( our_includes
  ../include/kernels.h
  ../include/libdpp_c.h
  common.hpp
  selector_base.hpp
  c_kernel.hpp
  l_kernel.hpp
  ../include/results.hpp
  ../include/fwddefs.hpp
  ../include/l_selection.hpp
  ../include/c_selector.hpp
  l_selection_impl.hpp
  c_selection_impl.hpp
)

include_directories(${libdpp_INCLUDES})

add_library(dpp_s SHARED ${sources} ${our_includes})
add_library(dpp STATIC ${sources} ${our_includes})