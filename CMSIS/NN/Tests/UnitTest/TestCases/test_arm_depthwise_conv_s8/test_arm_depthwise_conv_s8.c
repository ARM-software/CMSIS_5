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

#include "../Utils/validate.h"
#include "../TestData/basic/test_data.h"
#include "../TestData/stride2pad1/test_data.h"


// Not used
static const uint16_t dilation = 1;
static q15_t *bufferA = NULL;

void basic_arm_depthwise_conv_s8(void)
{
  arm_status expected = ARM_MATH_SUCCESS;
  q7_t output[BASIC_DST_SIZE] = {0};

  const uint16_t ch_mult =1;

  arm_status result = arm_depthwise_conv_s8(basic_input,
                                            BASIC_CONV_W,
                                            BASIC_CONV_H,
                                            BASIC_IN_CH,
                                            basic_weights,
                                            BASIC_OUT_CH,
                                            ch_mult,
                                            BASIC_FILTER_X,
                                            BASIC_FILTER_Y,
                                            BASIC_PAD_X,
                                            BASIC_PAD_Y,
                                            BASIC_STRIDE_X,
                                            BASIC_STRIDE_Y,
                                            basic_biases,
                                            output,
                                            basic_output_shift,
                                            basic_output_mult,
                                            BASIC_OUT_CONV_W,
                                            BASIC_OUT_CONV_H,
                                            BASIC_OUTPUT_OFFSET,
                                            BASIC_INPUT_OFFSET,
                                            BASIC_OUT_ACTIVATION_MIN,
                                            BASIC_OUT_ACTIVATION_MAX,
                                            dilation,
                                            dilation,
                                            bufferA);

  TEST_ASSERT_EQUAL(expected, result);
  TEST_ASSERT_TRUE(validate(output, basic_output_ref, BASIC_DST_SIZE));
}

void stride2pad1_arm_depthwise_conv_s8(void)
{
  arm_status expected = ARM_MATH_SUCCESS;
  q7_t output[STRIDE2PAD1_DST_SIZE] = {0};

  const uint16_t ch_mult =1;

  arm_status result = arm_depthwise_conv_s8(stride2pad1_input,
                                            STRIDE2PAD1_CONV_W,
                                            STRIDE2PAD1_CONV_H,
                                            STRIDE2PAD1_IN_CH,
                                            stride2pad1_weights,
                                            STRIDE2PAD1_OUT_CH,
                                            ch_mult,
                                            STRIDE2PAD1_FILTER_X,
                                            STRIDE2PAD1_FILTER_Y,
                                            STRIDE2PAD1_PAD_X,
                                            STRIDE2PAD1_PAD_Y,
                                            STRIDE2PAD1_STRIDE_X,
                                            STRIDE2PAD1_STRIDE_Y,
                                            stride2pad1_biases,
                                            output,
                                            stride2pad1_output_shift,
                                            stride2pad1_output_mult,
                                            STRIDE2PAD1_OUT_CONV_W,
                                            STRIDE2PAD1_OUT_CONV_H,
                                            STRIDE2PAD1_OUTPUT_OFFSET,
                                            STRIDE2PAD1_INPUT_OFFSET,
                                            STRIDE2PAD1_OUT_ACTIVATION_MIN,
                                            STRIDE2PAD1_OUT_ACTIVATION_MAX,
                                            dilation,
                                            dilation,
                                            bufferA);

  TEST_ASSERT_EQUAL(expected, result);
  TEST_ASSERT_TRUE(validate(output, stride2pad1_output_ref, STRIDE2PAD1_DST_SIZE));
}
