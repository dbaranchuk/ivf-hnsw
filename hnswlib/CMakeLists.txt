cmake_minimum_required (VERSION 2.8)

# hnswlib project
project(hnswlib C CXX)

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src)	# ivf-hnsw root directory

add_library(hnswlib STATIC ${headers} ${sources})
SET( CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
target_link_libraries(hnswlib)
