set( sources
  main.cpp
)

add_executable(word_sim
  ${sources}
)

link_directories(  
  ${libdpp_BINARY_DIR}
)

include_directories(  
  ${libdpp_INCLUDES}
)

target_link_libraries(word_sim  
  dpp
)
