cmake_minimum_required (VERSION 3.14)
include(CMakePrintHelpers)

project(Example1)


add_executable(example1 main.cpp)

sdf(example1)
add_sdf_dir(example1)

target_include_directories(example1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(example1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/generated)
