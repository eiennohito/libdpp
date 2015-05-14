set( sources
  main.cpp
  kernels.cpp
  selection_probability.cpp
)

add_executable(dpp_tests
  ${sources}
)

message(${libdpp_BINARY_DIR})

link_directories(
  ${gtest_LIBRARY_DIRECTORY}
  ${libdpp_BINARY_DIR}
)

include_directories(
  ../libs/gtest/include
  ${libdpp_INCLUDES}
)

target_link_libraries(dpp_tests
  gtest
  dpp
)
