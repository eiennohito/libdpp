set( sources
  kernels.cpp
  libdpp_c.cpp
)

set ( our_includes
  ../includes/kernels.h
  ../includes/libdpp_c.h
)

include_directories(${libdpp_INCLUDES})

add_library(dpp ${sources} ${includes})