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

#include "../TestData/mul/test_data.h"
#include "../Utils/validate.h"

void mul_arm_elementwise_mul_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    int8_t output[MUL_DST_SIZE] = {0};

    const int8_t *input_data1 = mul_input1;
    const int8_t *input_data2 = mul_input2;

    const int32_t input_1_offset = MUL_INPUT1_OFFSET;
    const int32_t input_2_offset = MUL_INPUT2_OFFSET;

    const int32_t out_offset = MUL_OUTPUT_OFFSET;
    const int32_t out_mult = MUL_OUTPUT_MULT;
    const int32_t out_shift = MUL_OUTPUT_SHIFT;

    const int32_t out_activation_min = MUL_OUT_ACTIVATION_MIN;
    const int32_t out_activation_max = MUL_OUT_ACTIVATION_MAX;

    arm_status result = arm_elementwise_mul_s8(input_data1,
                                               input_data2,
                                               input_1_offset,
                                               input_2_offset,
                                               output,
                                               out_offset,
                                               out_mult,
                                               out_shift,
                                               out_activation_min,
                                               out_activation_max,
                                               MUL_DST_SIZE);

    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, mul_output_ref, MUL_DST_SIZE));
}
