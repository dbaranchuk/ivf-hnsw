cmake_minimum_required (VERSION 2.8)

# ivf-hnsw project
project(ivf-hnsw C CXX)

option(BUILD_TEST "Build tests" ON)

include_directories("${PROJECT_BINARY_DIR}")

add_subdirectory(faiss)
add_subdirectory(hnswlib)

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# specify header and cpp files
file(GLOB ivf-hnsw_cpu_headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB ivf-hnsw_cpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(ivf-hnsw STATIC ${ivf-hnsw_cpu_headers} ${ivf-hnsw_cpu_cpp})

SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -m64 -g -msse4 -mpopcnt -Wno-sign-compare")
target_link_libraries(ivf-hnsw faiss hnswlib)

# build tests
#if(BUILD_TEST)
    add_subdirectory(tests)
#endif(BUILD_TEST)