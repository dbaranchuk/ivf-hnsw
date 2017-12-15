cmake_minimum_required (VERSION 2.8)

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
include_directories(../../)	# ivf-hnsw root directory
#include_directories(../)

SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source})
    target_link_libraries(${name} ${ivf-hnsw_lib}) #${faiss_lib}

    # Install
    install(TARGETS ${name} DESTINATION test)
endforeach(source)
