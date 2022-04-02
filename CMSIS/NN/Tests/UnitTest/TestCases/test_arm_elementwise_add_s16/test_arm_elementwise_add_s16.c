/*
 * Copyright (C) 2022 Arm Limited or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "arm_nnfunctions.h"
#include "unity.h"

#include "../TestData/add_s16/test_data.h"
#include "../Utils/validate.h"

void add_s16_arm_elementwise_add_s16(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    int16_t output[ADD_S16_DST_SIZE] = {0};

    const int16_t *input_data1 = add_s16_input1;
    const int16_t *input_data2 = add_s16_input2;

    const int32_t input_1_mult = ADD_S16_INPUT1_MULT;
    const int32_t input_1_shift = ADD_S16_INPUT1_SHIFT;
    const int32_t input_1_offset = ADD_S16_INPUT1_OFFSET;
    const int32_t input_2_mult = ADD_S16_INPUT2_MULT;
    const int32_t input_2_shift = ADD_S16_INPUT2_SHIFT;
    const int32_t input_2_offset = ADD_S16_INPUT2_OFFSET;

    const int32_t left_shift = ADD_S16_LEFT_SHIFT;

    const int32_t out_offset = ADD_S16_OUTPUT_OFFSET;
    const int32_t out_mult = ADD_S16_OUTPUT_MULT;
    const int32_t out_shift = ADD_S16_OUTPUT_SHIFT;

    const int32_t out_activation_min = ADD_S16_OUT_ACTIVATION_MIN;
    const int32_t out_activation_max = ADD_S16_OUT_ACTIVATION_MAX;

    arm_status result = arm_elementwise_add_s16(input_data1,
                                                input_data2,
                                                input_1_offset,
                                                input_1_mult,
                                                input_1_shift,
                                                input_2_offset,
                                                input_2_mult,
                                                input_2_shift,
                                                left_shift,
                                                output,
                                                out_offset,
                                                out_mult,
                                                out_shift,
                                                out_activation_min,
                                                out_activation_max,
                                                ADD_S16_DST_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate_s16(output, add_s16_output_ref, ADD_S16_DST_SIZE));
}
