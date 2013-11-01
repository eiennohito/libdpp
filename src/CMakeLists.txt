set( sources
  kernels.cpp
)

include_directories(${libdpp_INCLUDES})

add_library(dpp ${sources})