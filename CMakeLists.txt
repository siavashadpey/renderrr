cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0048 NEW)

project(Renderrr VERSION 0.1)
set(lib_name Renderrr)

option(CUDA "select wheter CUDA is used or not" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
if (CUDA)
	set(CUDA_ENABLED ON)
    SET(CUDA_SEPARABLE_COMPILATION ON)
	find_package(CUDA 10.1 REQUIRED)
	message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
    cuda_select_nvcc_arch_flags(ARCH_FLAGS)
    LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS} -Wno-deprecated-gpu-targets --expt-relaxed-constexpr -rdc=true)
    set(CUDA_PROPAGATE_HOST_FLAG OFF)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(driver)
