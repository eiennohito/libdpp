set( sources
  main.cpp
)

add_executable(dpp_tests
  ${sources}
)

link_directories(
  ${gtest_LIBRARY_DIRECTORY}
)

include_directories(
  ../libs/gtest/include
  ${libdpp_INCLUDES}
)

target_link_libraries(dpp_tests
  gtest
  dpp
)
