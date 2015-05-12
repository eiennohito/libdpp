set(sources
    sample.cpp
)

add_executable(
    mmap_sample_dpp
    ${sources}
)

link_directories(
  ${libdpp_BINARY_DIR}
)

target_link_libraries(mmap_sample_dpp
  dpp
)

add_executable(
    mmap_sample_dpp_full
    full_kernel.cpp
)

target_link_libraries(mmap_sample_dpp_full
  dpp
)

include_directories(
  ${libdpp_INCLUDES}
)