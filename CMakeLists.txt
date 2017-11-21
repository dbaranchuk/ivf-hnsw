cmake_minimum_required (VERSION 2.6)
project (ivf_hnsw_lib)

include_directories("${PROJECT_BINARY_DIR}")

project(IVF_HNSW)

set(SOURCE_EXE main.cpp)            

set(SOURCE_LIB1 hybrid.cpp)
set(SOURCE_LIB2 IndexIVF_HNSW_PQ.cpp)

add_library(ivf_hnsw STATIC ${SOURCE_LIB1} ${SOURCE_LIB2})

add_subdirectory(faiss)
add_subdirectory(hnswlib)

add_executable(main ${SOURCE_EXE})  
SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )


target_link_libraries(main ivf_hnsw faiss hnswlib)