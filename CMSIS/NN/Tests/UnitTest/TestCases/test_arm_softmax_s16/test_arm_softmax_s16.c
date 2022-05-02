/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
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

#include "unity.h"
#include <arm_nnfunctions.h>

#include "../Common/Softmax/exp_lut_data.h"
#include "../Common/Softmax/one_by_one_lut_data.h"
#include "../TestData/softmax_s16/test_data.h"
#include "../Utils/validate.h"

#define REPEAT_NUM (2)

void softmax_s16_arm_softmax_s16(void)
{
    const int32_t num_rows = SOFTMAX_S16_NUM_ROWS;
    const int32_t row_size = SOFTMAX_S16_ROW_SIZE;
    const int32_t mult = SOFTMAX_S16_INPUT_MULT;
    const int32_t shift = SOFTMAX_S16_INPUT_LEFT_SHIFT;
    const q15_t *input_data = softmax_s16_input;
    const cmsis_nn_softmax_lut_s16 softmax_params = {.exp_lut = softmax_s16_exp_lut,
                                                     .one_by_one_lut = softmax_s16_one_by_one_lut};
    int16_t output[SOFTMAX_S16_DST_SIZE];

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_softmax_s16(input_data, num_rows, row_size, mult, shift, &softmax_params, output);
        TEST_ASSERT_TRUE(validate_s16(output, softmax_s16_output_ref, SOFTMAX_S16_DST_SIZE));
    }
}
