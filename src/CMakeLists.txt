set( sources
  kernels.cpp
)

set ( our_includes
  ../includes/kernels.h
)

include_directories(${libdpp_INCLUDES})

add_library(dpp ${sources} ${includes})