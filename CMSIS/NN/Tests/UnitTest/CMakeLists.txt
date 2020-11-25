#
# Copyright (c) 2019-2020 Arm Limited. All rights reserved.
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

set_property(GLOBAL PROPERTY cmsis_nn_unit_test_executables)

# This function should be used instead of add_executable.
function(add_cmsis_nn_unit_test_executable)
    get_property(tmp GLOBAL PROPERTY cmsis_nn_unit_test_executables)
    foreach(target ${ARGV})
        set(tmp "${tmp} ${target}")
        add_executable(${target})
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_options(${target} PRIVATE "--specs=nosys.specs")
        endif()
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

