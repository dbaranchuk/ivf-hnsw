cmake_minimum_required (VERSION 2.8)

# TODO Добавить cpp для сборки

# hnswlib project
project(hnswlib C CXX)

file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
include_directories(../../)	# ivf-hnsw root directory

add_library(hnswlib STATIC ${headers} ${sources})

target_link_libraries(hnswlib)