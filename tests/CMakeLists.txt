# Build each source file independently
include_directories(../../)	# ivf-hnsw root directory

foreach(source ${srcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source})
    target_link_libraries(${name} ${faiss_lib})

    # Install
    install(TARGETS ${name} DESTINATION test)
endforeach(source)