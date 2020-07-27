include_directories(${project_include})

if (CUDA_ENABLED)
    set_source_files_properties(three_balls.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set_source_files_properties(many_balls.cpp  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

    cuda_add_executable(three_balls three_balls.cpp)
    cuda_add_executable(many_balls   many_balls.cpp)

    target_link_libraries(three_balls ${lib_name} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
    target_link_libraries(many_balls  ${lib_name} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
else()
    add_executable(three_balls three_balls.cpp)
    add_executable(many_balls   many_balls.cpp)

    target_link_libraries(three_balls ${lib_name})
    target_link_libraries(many_balls  ${lib_name})
endif()