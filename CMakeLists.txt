# The name of our project is "HELLO". CMakeLists files in this project can 
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and 
# to the root binary directory of the project as ${HELLO_BINARY_DIR}. 
cmake_minimum_required (VERSION 2.6) 
project (libdpp) 

# Recurse into the "Hello" and "Demo" subdirectories. This does not actually 
# cause another cmake executable to run. The same process will walk through 
# the project's entire directory structure. 


set(libdpp_INCLUDES ${libdpp_SOURCE_DIR}/include)

#needs C++11 to build
add_definitions("-std=c++11")

include_directories(
  ./libs/
)

add_subdirectory (src)
add_subdirectory (libs/gtest) 

add_subdirectory (test)

find_package(Boost COMPONENTS program_options)

if (${Boost_FOUND})
  include_directories(${Boost_INCLUDE_DIRS})
  add_subdirectory(word_similarity)
  add_subdirectory(mmap_sample)
endif()
