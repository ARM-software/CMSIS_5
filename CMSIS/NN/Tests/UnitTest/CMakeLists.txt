#
# Copyright (c) 2019-2022 Arm Limited.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.15.6)

project(cmsis_nn_unit_tests VERSION 0.0.1)

set(CMSIS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")

add_compile_options(-Ofast
                    -fomit-frame-pointer
                    -Werror
                    -Wimplicit-function-declaration
                    -Wunused-variable
                    -Wunused-function
                    -Wno-redundant-decls)

option(BUILD_CMSIS_NN_UNIT "If building the unit tests from another project, i.e. \
platform dependencies need to be provided externally." OFF)

if(NOT BUILD_CMSIS_NN_UNIT)
    set(BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300 ON)
else()
    set(BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300 OFF)
endif()

if(BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300)
    set(FVP_CORSTONE_300_PATH "${CMAKE_CURRENT_SOURCE_DIR}/Corstone-300" CACHE PATH
        "Dependencies for using FVP based on Arm Corstone-300 software.")
    set(CMAKE_EXECUTABLE_SUFFIX ".elf")
endif()

# Build the functions to be tested.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cmsis-nn)

# Target for all unit tests.
add_custom_target(cmsis_nn_unit_tests)

# This function should be used instead of add_executable.
set_property(GLOBAL PROPERTY cmsis_nn_unit_test_executables)
function(add_cmsis_nn_unit_test_executable)
    get_property(tmp GLOBAL PROPERTY cmsis_nn_unit_test_executables)
    foreach(target ${ARGV})
        set(tmp "${tmp} ${target}")
        add_executable(${target})
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_options(${target} PRIVATE "--specs=nosys.specs")
        endif()
        add_dependencies(cmsis_nn_unit_tests ${target})
    endforeach()
    set_property(GLOBAL PROPERTY cmsis_nn_unit_test_executables "${tmp}")
endfunction(add_cmsis_nn_unit_test_executable)

add_subdirectory(TestCases/test_arm_avgpool_s16)
add_subdirectory(TestCases/test_arm_avgpool_s8)
add_subdirectory(TestCases/test_arm_convolve_1x1_s8_fast)
add_subdirectory(TestCases/test_arm_convolve_fast_s16)
add_subdirectory(TestCases/test_arm_convolve_s16)
add_subdirectory(TestCases/test_arm_convolve_s8)
add_subdirectory(TestCases/test_arm_depthwise_conv_3x3_s8)
add_subdirectory(TestCases/test_arm_depthwise_conv_s16)
add_subdirectory(TestCases/test_arm_depthwise_conv_s8)
add_subdirectory(TestCases/test_arm_depthwise_conv_s8_opt)
add_subdirectory(TestCases/test_arm_elementwise_add_s16)
add_subdirectory(TestCases/test_arm_elementwise_add_s8)
add_subdirectory(TestCases/test_arm_elementwise_mul_s16)
add_subdirectory(TestCases/test_arm_elementwise_mul_s8)
add_subdirectory(TestCases/test_arm_fully_connected_s16)
add_subdirectory(TestCases/test_arm_fully_connected_s8)
add_subdirectory(TestCases/test_arm_max_pool_s16)
add_subdirectory(TestCases/test_arm_max_pool_s8)
add_subdirectory(TestCases/test_arm_softmax_s8)
add_subdirectory(TestCases/test_arm_softmax_s8_s16)
add_subdirectory(TestCases/test_arm_softmax_s16)
add_subdirectory(TestCases/test_arm_svdf_state_s16_s8)

set(MAKE_CMD "python3")
set(MAKE_CMD_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/unittest_targets.py")
set(MAKE_CMD_SCRIPT_OPTION "--download-and-generate-test-runners")
MESSAGE(STATUS "Downloading Unity and generating test runners for CMSIS-NN unit tests if needed..")
execute_process(COMMAND ${MAKE_CMD} ${MAKE_CMD_SCRIPT} ${MAKE_CMD_SCRIPT_OPTION}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(Unity)

# Link common dependencies.
get_property(executables GLOBAL PROPERTY cmsis_nn_unit_test_executables)
string(REPLACE " " ";" cmsis_nn_unit_test_list_of_executables ${executables})
foreach(target ${cmsis_nn_unit_test_list_of_executables})
    target_link_libraries(${target} LINK_PUBLIC unity)
    target_link_libraries(${target} LINK_PUBLIC cmsis-nn)
endforeach()

if(BUILD_CMSIS_NN_UNIT_TESTS_FOR_FVP_BASED_CORSTONE_300)
    add_library(retarget STATIC
        ${FVP_CORSTONE_300_PATH}/retarget.c
        ${FVP_CORSTONE_300_PATH}/uart.c)

    # Build CMSIS startup dependencies based on TARGET_CPU.
    string(REGEX REPLACE "^cortex-m([0-9]+)$" "ARMCM\\1" ARM_CPU ${CMAKE_SYSTEM_PROCESSOR})
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "cortex-m33")
        set(ARM_FEATURES "_DSP_FP")
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "cortex-m4")
        set(ARM_FEATURES "_FP")
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "cortex-m7")
        set(ARM_FEATURES "_DP")
    else()
        set(ARM_FEATURES "")
    endif()
    add_library(cmsis_startup STATIC)
    target_sources(cmsis_startup PRIVATE
        ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/startup_${ARM_CPU}.c
        ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/system_${ARM_CPU}.c)
    target_include_directories(cmsis_startup PUBLIC
        ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Include
        ${CMSIS_PATH}/CMSIS/Core/Include)
    target_compile_options(cmsis_startup INTERFACE -include${ARM_CPU}${ARM_FEATURES}.h)
    target_compile_definitions(cmsis_startup PRIVATE ${ARM_CPU}${ARM_FEATURES})

    # Linker file settings.
    set(LINK_FILE "${FVP_CORSTONE_300_PATH}/linker" CACHE PATH "Linker file.")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "ARMClang")
        set(LINK_FILE "${FVP_CORSTONE_300_PATH}/linker.scatter")
        set(LINK_FILE_OPTION "--scatter")
        set(LINK_ENTRY_OPTION "--entry")
        set(LINK_ENTRY "Reset_Handler")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(LINK_FILE "${FVP_CORSTONE_300_PATH}/linker.ld")
        set(LINK_FILE_OPTION "-T")
        set(LINK_ENTRY_OPTION "")
        set(LINK_ENTRY "")
    endif()

    # Link in FVP dependencies to every unit test.
    get_property(executables GLOBAL PROPERTY cmsis_nn_unit_test_executables)
    string(REPLACE " " ";" cmsis_nn_unit_test_list_of_executables ${executables})
    foreach(target ${cmsis_nn_unit_test_list_of_executables})
        target_link_libraries(${target} PRIVATE retarget)
        target_link_libraries(${target} PRIVATE $<TARGET_OBJECTS:cmsis_startup> cmsis_startup)

        add_dependencies(${target} retarget cmsis_startup)

        target_compile_definitions(${target} PUBLIC USING_FVP_CORSTONE_300)

        target_link_options(${target} PRIVATE ${LINK_FILE_OPTION} ${LINK_FILE} ${LINK_ENTRY_OPTION} ${LINK_ENTRY})
        set_target_properties(${target} PROPERTIES LINK_DEPENDS ${LINK_FILE})
    endforeach()
endif()
