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
#include "../TestData/maxpooling/test_data.h"


void maxpooling_arm_max_pool_s8_opt(void)
{
  q7_t output[MAXPOOLING_DST_SIZE] = {0};
  const arm_status expected = ARM_MATH_SUCCESS;

  arm_status result = arm_max_pool_s8_opt(MAXPOOLING_INPUT_H,
                                          MAXPOOLING_INPUT_W,
                                          MAXPOOLING_OUTPUT_H,
                                          MAXPOOLING_OUTPUT_W,
                                          MAXPOOLING_STRIDE_Y,
                                          MAXPOOLING_STRIDE_X,
                                          MAXPOOLING_FILTER_Y,
                                          MAXPOOLING_FILTER_X,
                                          MAXPOOLING_PAD_Y,
                                          MAXPOOLING_PAD_X,
                                          MAXPOOLING_OUT_ACTIVATION_MIN,
                                          MAXPOOLING_OUT_ACTIVATION_MAX,
                                          MAXPOOLING_IN_CH,
                                          maxpooling_input,
                                          NULL,
                                          output);
  TEST_ASSERT_EQUAL(expected, result);
  TEST_ASSERT_TRUE(validate(output, maxpooling_output_ref, MAXPOOLING_DST_SIZE));
}
