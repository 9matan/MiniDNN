cmake_minimum_required(VERSION 3.12)
project(MiniDNN C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package( Eigen3 REQUIRED )

set(MINIDNN_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
set(MINIDNN_SYSTEM_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
file(GLOB_RECURSE MINIDNN_HEADERS ${MINIDNN_INCLUDE_DIRS}/*.h)

option(DEBUG_ADD_MINIDNN_FILES_TO_TESTS "Add minidnn headers to tests sources. It is helpful for debugging" OFF)

add_subdirectory(tutorials/test1)
add_subdirectory(tutorials/test2)