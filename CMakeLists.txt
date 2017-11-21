cmake_minimum_required (VERSION 2.6)
project(IVF_HNSW)

include_directories("${PROJECT_BINARY_DIR}")

set(SOURCE_EXE main.cpp)            

set(SOURCE_LIB hybrid.cpp)

add_library(test STATIC ${SOURCE_LIB})

set(HEARDER_LIB IndexIVF_HNSW_PQ.h)
set(SOURCE_LIB IndexIVF_HNSW_PQ.cpp)
add_library(ivf_hnsw STATIC ${HEADER_LIB} ${SOURCE_LIB})

add_subdirectory(faiss)

add_executable(main ${SOURCE_EXE})  
SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

target_link_libraries(main ivf_hnsw test faiss)