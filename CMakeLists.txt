cmake_minimum_required (VERSION 2.8)

# ivf-hnsw project
project(ivf-hnsw C CXX)

add_subdirectory(faiss)
add_subdirectory(hnswlib)

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

file(GLOB ivfhnsw_src ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB ivfhnsw_include ${PROJECT_SOURCE_DIR}/include/*.h)

SET(CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")

add_library(ivfhnsw STATIC ${ivfhnsw_src})
link_libraries(ivfhnsw faiss hnswlib)


