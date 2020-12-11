# Copyright (C) 2010-2020 Arm Limited or its affiliates. All rights reserved.
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

# Extract the CPU number from the system processor
string(REGEX MATCH "^cortex-m([0-9]+)$" CPU_NUMBER ${CMAKE_SYSTEM_PROCESSOR})
if(NOT CPU_NUMBER)
    message(FATAL_ERROR "System processor '${CMAKE_SYSTEM_PROCESSOR}' not supported. Should be cortex-m<nr>.")
endif()
string(REGEX REPLACE "^cortex-m([0-9]+)$" "\\1" CPU_NUMBER ${CMAKE_SYSTEM_PROCESSOR})

set(ARM_CPU "ARMCM${CPU_NUMBER}")

# Set CPU specific features
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "cortex-m33")
    set(ARM_FEATURES "_DSP_FP")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "cortex-m4")
    set(ARM_FEATURES "_FP")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "cortex-m7")
    set(ARM_FEATURES "_DP")
else()
    set(ARM_FEATURES "")
endif()

# CMSIS core
add_library(cmsis_core INTERFACE)
target_include_directories(cmsis_core INTERFACE ${CMSIS_PATH}/CMSIS/Core/Include)

# CMSIS device
add_library(cmsis_device INTERFACE)
target_include_directories(cmsis_device INTERFACE ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Include)
target_compile_options(cmsis_device INTERFACE -include${ARM_CPU}${ARM_FEATURES}.h)

# CMSIS startup
add_library(cmsis_startup STATIC
    ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/startup_${ARM_CPU}.c
    ${CMSIS_PATH}/Device/ARM/${ARM_CPU}/Source/system_${ARM_CPU}.c)
target_compile_definitions(cmsis_startup PRIVATE ${ARM_CPU}${ARM_FEATURES})
target_link_libraries(cmsis_startup PRIVATE cmsis_device cmsis_core)
