cmake_minimum_required (VERSION 2.8)

# TODO Добавить cpp для сборки

# hnswlib project
project(hnswlib C CXX)

SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )


file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
include_directories(../../)	# ivf-hnsw root directory

add_library(hnswlib STATIC ${headers} ${sources})

target_link_libraries(hnswlib)