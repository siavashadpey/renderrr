cmake_minimum_required(VERSION 2.8.11)
cmake_policy(SET CMP0048 NEW)
project(Renderrr VERSION 0.1)
set(lib_name Renderrr)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(GNU_WARNINGS "-Wall -Wextra -Wnon-virtual-dtor -Wpedantic -Wshadow")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GNU_WARNINGS}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set(project_include ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(driver)