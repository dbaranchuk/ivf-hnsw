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
target_link_libraries(ivfhnsw faiss hnswlib)

FIND_PACKAGE(SWIG)
FIND_PACKAGE(PythonLibs)
if(SWIG_FOUND AND PythonLibs_FOUND)
  INCLUDE(${SWIG_USE_FILE})
  INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

  file(DOWNLOAD https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i ./numpy.i)
  file(DOWNLOAD https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/pyfragments.swg ./pyfragments.swg)

  set(CMAKE_SWIG_OUTDIR ${CMAKE_BINARY_DIR}/lib)
  set(SWIG_FEATURES "-Iinclude")
  file(GLOB swig_interface interface/wrapper.i)

  SET_SOURCE_FILES_PROPERTIES(${swig_interface} PROPERTIES CPLUSPLUS ON)

  swig_add_module(wrapper python ${swig_interface} ${ivfhnsw_src})
  swig_link_libraries(wrapper faiss hnswlib ${PYTHON_LIBRARIES})

endif()

add_subdirectory(tests)
