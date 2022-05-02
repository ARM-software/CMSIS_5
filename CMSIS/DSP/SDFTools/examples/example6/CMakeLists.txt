cmake_minimum_required (VERSION 3.14)
include(CMakePrintHelpers)

project(Example6)


add_executable(example6 main.cpp mfccConfigData.c)

sdf(example6)
add_sdf_dir(example6)

target_include_directories(example6 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(example6 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/generated)

target_link_libraries(example6 PRIVATE CMSISDSP)

file(COPY input_example6.txt DESTINATION ${CMAKE_BINARY_DIR})