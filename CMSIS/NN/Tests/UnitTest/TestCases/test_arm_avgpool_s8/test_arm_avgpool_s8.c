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
#include "../TestData/avgpooling/test_data.h"


void avgpooling_arm_avgpool_s8(void)
{
  q7_t output[AVGPOOLING_DST_SIZE] = {0};
  const arm_status expected = ARM_MATH_SUCCESS;
  const int32_t buf_size = arm_avgpool_s8_get_buffer_size(AVGPOOLING_INPUT_W, AVGPOOLING_IN_CH);
  q15_t *bufferA = (q15_t*)malloc(buf_size);

  arm_status result = arm_avgpool_s8(AVGPOOLING_INPUT_H,
                                     AVGPOOLING_INPUT_W,
                                     AVGPOOLING_OUTPUT_H,
                                     AVGPOOLING_OUTPUT_W,
                                     AVGPOOLING_STRIDE_Y,
                                     AVGPOOLING_STRIDE_X,
                                     AVGPOOLING_FILTER_Y,
                                     AVGPOOLING_FILTER_X,
                                     AVGPOOLING_PAD_Y,
                                     AVGPOOLING_PAD_X,
                                     AVGPOOLING_OUT_ACTIVATION_MIN,
                                     AVGPOOLING_OUT_ACTIVATION_MAX,
                                     AVGPOOLING_IN_CH,
                                     avgpooling_input,
                                     bufferA,
                                     output);
  free(bufferA);
  TEST_ASSERT_EQUAL(expected, result);
  TEST_ASSERT_TRUE(validate(output, avgpooling_output_ref, AVGPOOLING_DST_SIZE));
}
