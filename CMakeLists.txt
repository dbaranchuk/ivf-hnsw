cmake_minimum_required (VERSION 2.8)

# ivf-hnsw project
project(ivf-hnsw C CXX)

include_directories("${PROJECT_BINARY_DIR}")

add_subdirectory(faiss)
add_subdirectory(hnswlib)

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_SWIG_OUTDIR ${CMAKE_BINARY_DIR}/lib)

# specify header and cpp files
file(GLOB ivf-hnsw_cpu_headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB ivf-hnsw_cpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR} ${PYTHON_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

file(DOWNLOAD https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i ./numpy.i)
file(DOWNLOAD https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/pyfragments.swg ./pyfragments.swg)

SET_SOURCE_FILES_PROPERTIES(ivfhnsw.i PROPERTIES CPLUSPLUS ON)

SET(CMAKE_SWIG_FLAGS  "")
SET(CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")

swig_add_module(ivfhnsw python ivfhnsw.i IndexIVF_HNSW.cpp)
swig_link_libraries(ivfhnsw faiss hnswlib ${PYTHON_LIBRARIES})


