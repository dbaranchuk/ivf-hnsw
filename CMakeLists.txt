cmake_minimum_required (VERSION 2.6)
project (hnsw_lib)

include_directories("${PROJECT_BINARY_DIR}")



project(hello_world)                

set(SOURCE_EXE main.cpp)            

set(SOURCE_LIB1 deep_10m.cpp)
set(SOURCE_LIB2 sift_1b.cpp)

add_library(deep_test STATIC ${SOURCE_LIB1})
add_library(sift_test STATIC ${SOURCE_LIB2})

add_executable(main ${SOURCE_EXE})  
SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

#target_link_libraries(main sift_test)
target_link_libraries(main deep_test)