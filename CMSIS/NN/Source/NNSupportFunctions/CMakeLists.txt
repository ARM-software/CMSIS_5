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

project(CMSISNNSupport)

file(GLOB SRC "./*_s8.c")
add_library(CMSISNNSupport STATIC ${SRC})
target_sources(CMSISNNSupport PUBLIC arm_q7_to_q15_with_offset.c)

### Includes
target_include_directories(CMSISNNSupport PUBLIC "${NN}/Include")
target_include_directories(CMSISNNSupport PUBLIC "${ROOT}/CMSIS/Core/Include")
target_include_directories(CMSISNNSupport PUBLIC "${ROOT}/CMSIS/DSP/Include")
