cmake_minimum_required (VERSION 2.8)

file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# Build each source file independently
include_directories(../../)	# ivf-hnsw root directory

add_library(hnswlib STATIC ${headers})

target_link_libraries(hnswlib)