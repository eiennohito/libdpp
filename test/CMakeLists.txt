set( sources
  
)

add_executable(dpp_tests
  ${gtest_LIBRARY}
  ${dpp}
  ${sources}
)