cmake_minimum_required (VERSION 2.8)
project (ivf_hnsw_lib)

option(BUILD_TEST "Build tests" ON)

include_directories("${PROJECT_BINARY_DIR}")

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#set(SOURCE_EXE demo_deep1b.cpp)

set(SOURCE_LIB_h IndexIVF_HNSW.h)
set(SOURCE_LIB_cpp IndexIVF_HNSW.cpp)
set(SOURCE_LIB_uh utils.h)
set(SOURCE_LIB_ucpp utils.cpp)

set(ivf-hnsw_lib ivf-hnsw)
add_library(${ivf-hnsw_lib} STATIC ${SOURCE_LIB_h} ${SOURCE_LIB_cpp} ${SOURCE_LIB_uh} ${SOURCE_LIB_ucpp})

add_subdirectory(faiss)

#add_executable(demo_deep1b ${SOURCE_EXE})

SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

target_link_libraries(${ivf-hnsw_lib} ${faiss_lib}) #demo_deep1b

# build tests
#if(BUILD_TEST)
    add_subdirectory(tests)
#endif(BUILD_TEST)