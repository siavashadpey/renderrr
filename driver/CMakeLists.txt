include_directories(${project_include})

add_executable(main main.cpp)
target_link_libraries(main ${lib_name})
