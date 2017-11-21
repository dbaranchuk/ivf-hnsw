cmake_minimum_required (VERSION 2.6)
project (hnsw)

set(SOURCE_LIB1 hnswalg.h)
set(SOURCE_LIB2 hnswlib.h)
set(SOURCE_LIB3 L2space.h)
set(SOURCE_LIB4 visited_list_pool.h)

add_library(hnsw STATIC ${SOURCE_LIB1} ${SOURCE_LIB2} ${SOURCE_LIB3} ${SOURCE_LIB4})

SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

set(hnsw_lib hnsw)
target_link_libraries(${hnsw_lib})