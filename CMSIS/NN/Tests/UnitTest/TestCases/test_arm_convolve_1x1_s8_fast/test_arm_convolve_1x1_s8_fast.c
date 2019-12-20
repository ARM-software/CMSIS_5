/*
 * Copyright (C) 2010-2020 Arm Limited or its affiliates. All rights reserved.
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

#include <arm_nnfunctions.h>
#include <stdlib.h>

#include "../Utils/validate.h"
#include "../TestData/kernel1x1/test_data.h"

void kernel1x1_arm_convolve_1x1_s8_fast(void)
{
  arm_status expected = ARM_MATH_SUCCESS;
  q7_t output[KERNEL1X1_DST_SIZE] = {0};
  const int32_t buf_size = arm_convolve_1x1_s8_fast_get_buffer_size(KERNEL1X1_IN_CH);
  q15_t *bufferA = (q15_t*)malloc(buf_size);

  arm_status result =  arm_convolve_1x1_s8_fast(kernel1x1_input,
                                                KERNEL1X1_CONV_W,
                                                KERNEL1X1_CONV_H,
                                                KERNEL1X1_IN_CH,
                                                KERNEL1X1_INPUT_BATCHES,
                                                kernel1x1_weights,
                                                KERNEL1X1_OUT_CH,
                                                KERNEL1X1_PAD_X,
                                                KERNEL1X1_PAD_Y,
                                                KERNEL1X1_STRIDE_X,
                                                KERNEL1X1_STRIDE_Y,
                                                kernel1x1_biases,
                                                output,
                                                kernel1x1_output_shift,
                                                kernel1x1_output_mult,
                                                KERNEL1X1_OUTPUT_OFFSET,
                                                KERNEL1X1_INPUT_OFFSET,
                                                KERNEL1X1_OUT_ACTIVATION_MIN,
                                                KERNEL1X1_OUT_ACTIVATION_MAX,
                                                KERNEL1X1_OUT_CONV_W,
                                                KERNEL1X1_OUT_CONV_H,
                                                bufferA);

  TEST_ASSERT_EQUAL(expected, result);
  TEST_ASSERT_TRUE(validate(output, kernel1x1_output_ref, KERNEL1X1_DST_SIZE));

  free(bufferA);
}
