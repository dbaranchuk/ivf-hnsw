cmake_minimum_required (VERSION 2.8)

file(GLOB srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build each source file independently
include_directories(../../)	# ivf-hnsw root directory

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source})
    target_link_libraries(${name} ivf-hnsw faiss)

    # Install
    install(TARGETS ${name} DESTINATION test)
endforeach(source)
