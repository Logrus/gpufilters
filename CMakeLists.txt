cmake_minimum_required(VERSION 2.8.3)

project(GPUFILTERS)
find_package(CUDA QUIET REQUIRED)

# The version number.
set (GPUFILTERS_VERSION_MAJOR 1)
set (GPUFILTERS_VERSION_MINOR 0)

include(cmake/Utils.cmake REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#link_directories(${CMAKE_BINARY_DIR})

include_directories("${PROJECT_SOURCE_DIR}/src" "${CUDA_TOOLKIT_INCLUDE}")
add_subdirectory(src)
add_subdirectory(apps)
