#
# Copyright (c) 2019-2021 Arm Limited. All rights reserved.
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

set(CMSIS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." CACHE PATH "Path to CMSIS.")

# External dependencies.
set(USE_ETHOSU_CORE_PLATFORM_DEFAULT OFF)
set(ETHOSU_CORE_PLATFORM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../core_platform" CACHE PATH "Path to Arm Ethos-U Core Platform project.")
set(FVP_CORSTONE_300_PATH "${ETHOSU_CORE_PLATFORM_PATH}/targets/corstone-300")

option(USE_ETHOSU_CORE_PLATFORM "Use Arm Ethos-U Core Platform project." ${USE_ETHOSU_CORE_PLATFORM_DEFAULT})

if(USE_ETHOSU_CORE_PLATFORM)
    set(CMAKE_EXECUTABLE_SUFFIX ".elf")
endif()

set_property(GLOBAL PROPERTY cmsis_nn_unit_test_executables)

# Target for all unit tests.
add_custom_target(cmsis_nn_unit_tests)

# This function should be used instead of add_executable.
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

add_library(arm_nn_mat_mult_core)
target_include_directories(arm_nn_mat_mult_core PRIVATE ${CMSIS_PATH}/CMSIS/NN/Include)
target_include_directories(arm_nn_mat_mult_core PRIVATE ${CMSIS_PATH}/CMSIS/DSP/Include)
target_include_directories(arm_nn_mat_mult_core PRIVATE ${CMSIS_PATH}/CMSIS/Core/Include)
target_sources(arm_nn_mat_mult_core PRIVATE
    ${CMSIS_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c
    ${CMSIS_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c)

add_library(arm_nn_depthwise_conv_nt)
target_include_directories(arm_nn_depthwise_conv_nt PRIVATE ${CMSIS_PATH}/CMSIS/NN/Include)
target_include_directories(arm_nn_depthwise_conv_nt PRIVATE ${CMSIS_PATH}/CMSIS/DSP/Include)
target_include_directories(arm_nn_depthwise_conv_nt PRIVATE ${CMSIS_PATH}/CMSIS/Core/Include)
target_sources(arm_nn_depthwise_conv_nt PRIVATE
    ${CMSIS_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c
    ${CMSIS_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.c)

add_subdirectory(TestCases/test_arm_avgpool_s8)
add_subdirectory(TestCases/test_arm_convolve_1x1_s8_fast)
add_subdirectory(TestCases/test_arm_convolve_s8)
add_subdirectory(TestCases/test_arm_depthwise_conv_3x3_s8)
add_subdirectory(TestCases/test_arm_depthwise_conv_s8)
add_subdirectory(TestCases/test_arm_depthwise_conv_s8_opt)
add_subdirectory(TestCases/test_arm_fully_connected_s8)
add_subdirectory(TestCases/test_arm_max_pool_s8)

set(MAKE_CMD "python3")
set(MAKE_CMD_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/unittest_targets.py")
set(MAKE_CMD_SCRIPT_OPTION "--download-and-generate-test-runners")
MESSAGE(STATUS "Downloading Unity and generating test runners for CMSIS-NN unit tests if needed..")
execute_process(COMMAND ${MAKE_CMD} ${MAKE_CMD_SCRIPT} ${MAKE_CMD_SCRIPT_OPTION}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(Unity)

if(USE_ETHOSU_CORE_PLATFORM)
add_subdirectory(${FVP_CORSTONE_300_PATH} fvp_corstone_300)

add_library(retarget STATIC
                     ${FVP_CORSTONE_300_PATH}/retarget.c
                     ${FVP_CORSTONE_300_PATH}/uart.c)

# Link in dependencies to every unit test.
get_property(executables GLOBAL PROPERTY cmsis_nn_unit_test_executables)
string(REPLACE " " ";" cmsis_nn_unit_test_list_of_executables ${executables})
foreach(target ${cmsis_nn_unit_test_list_of_executables})
    target_link_libraries(${target} PRIVATE $<TARGET_OBJECTS:retarget> retarget)
    target_link_libraries(${target} PRIVATE $<TARGET_OBJECTS:cmsis_startup> cmsis_device)
    add_dependencies(${target} retarget cmsis_startup)

    target_compile_definitions(${target} PUBLIC USING_FVP_CORSTONE_300)
    target_link_options(${target} PRIVATE --entry Reset_Handler)

    # Copy linker files.
    if (CMAKE_CXX_COMPILER_ID STREQUAL "ARMClang")
        configure_file(${FVP_CORSTONE_300_PATH}/platform.scatter
        ${CMAKE_CURRENT_SOURCE_DIR}/platform_${target}.scatter COPYONLY)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        configure_file(${FVP_CORSTONE_300_PATH}/platform.ld
        ${CMAKE_CURRENT_SOURCE_DIR}/platform_${target}.ld COPYONLY)
    endif()

    ethosu_link_options(${target} PRIVATE LINK_FILE platform_${target})
endforeach()
endif()
