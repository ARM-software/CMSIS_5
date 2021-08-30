cmake_minimum_required (VERSION 3.14)
include(CMakePrintHelpers)

project(Example2)


add_executable(example2 main.cpp)

sdf(example2)
add_sdf_dir(example2)

target_include_directories(example2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(example2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/generated)

target_link_libraries(example2 PRIVATE CMSISDSP)