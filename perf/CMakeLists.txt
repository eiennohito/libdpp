set(sources
  perf_run.cpp
)

add_executable(dpp_perf ${sources} perf_common.hpp measure_eigen.hpp measure_eigen.cpp)

target_link_libraries(dpp_perf dpp)

link_directories(
  ${libdpp_BINARY_DIR}
)

include_directories(
  ${libdpp_INCLUDES}
)