cmake_minimum_required (VERSION 2.6)
project (hnsw_lib)

include_directories("${PROJECT_BINARY_DIR}")


project(HNSW)

set(SOURCE_EXE main.cpp)            

set(SOURCE_LIB1 hybrid.cpp)
set(SOURCE_LIB2 IndexIVF_HNSW_PQ.cpp)

#file(GLOB faiss_cpu_headers *.h)
#file(GLOB faiss_cpu_cpp *.cpp)
#set(faiss_lib faiss)
#add_library(${faiss_lib} STATIC ${faiss_cpu_headers} ${faiss_cpu_cpp})

add_library(hybrid_test STATIC ${SOURCE_LIB1}) #${SOURCE_LIB2})

add_subdirectory(faiss)

add_executable(main ${SOURCE_EXE})  
SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )


target_link_libraries(main hybrid_test faiss)