cmake_minimum_required (VERSION 2.8)

# ivf-hnsw project
project(ivf-hnsw C CXX)

add_subdirectory(faiss)
add_subdirectory(hnswlib)

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_SWIG_OUTDIR ${CMAKE_BINARY_DIR})

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include ${PYTHON_INCLUDE_PATH})

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*)
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/include/*)
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/interface/*)

file(DOWNLOAD https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i ./numpy.i)
file(DOWNLOAD https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/pyfragments.swg ./pyfragments.swg)

SET(CMAKE_CXX_FLAGS "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")
SET(SWIG_FEATURES "-Iinclude")
SET_SOURCE_FILES_PROPERTIES(interface/ivfhnsw.i PROPERTIES CPLUSPLUS ON)

swig_add_module(ivfhnsw python interface/ivfhnsw.i src/IndexIVF_HNSW.cpp)
swig_link_libraries(ivfhnsw faiss hnswlib ${PYTHON_LIBRARIES})


