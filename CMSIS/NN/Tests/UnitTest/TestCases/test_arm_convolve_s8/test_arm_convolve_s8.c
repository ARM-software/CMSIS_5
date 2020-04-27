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
#include "../TestData/basic/test_data.h"
#include "../TestData/stride2pad1/test_data.h"
#include "../TestData/conv_2/test_data.h"
#include "../TestData/conv_3/test_data.h"
#include "../TestData/conv_1_x_n_1/test_data.h"
#include "../TestData/conv_1_x_n_2/test_data.h"
#include "../TestData/conv_1_x_n_3/test_data.h"

void basic_arm_convolve_s8(void)
{
  const arm_status expected = ARM_MATH_SUCCESS;
  q7_t output[BASIC_DST_SIZE] = {0};

  const int32_t buf_size = arm_convolve_s8_get_buffer_size(BASIC_IN_CH, BASIC_FILTER_X, BASIC_FILTER_Y);
  q15_t *bufferA = (q15_t*)malloc(buf_size);

  arm_status result = arm_convolve_s8(basic_input,
                                      BASIC_INPUT_W,
                                      BASIC_INPUT_H,
                                      BASIC_IN_CH,
                                      BASIC_INPUT_BATCHES,
                                      basic_weights,
                                      BASIC_OUT_CH,
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
                                      BASIC_OUTPUT_OFFSET,
                                      BASIC_INPUT_OFFSET,
                                      BASIC_OUT_ACTIVATION_MIN,
                                      BASIC_OUT_ACTIVATION_MAX,
                                      BASIC_OUTPUT_W,
                                      BASIC_OUTPUT_H,
                                      bufferA);

  TEST_ASSERT_EQUAL(expected, result);
  TEST_ASSERT_TRUE(validate(output, basic_output_ref, BASIC_DST_SIZE));

  free(bufferA);
}


void stride2pad1_arm_convolve_s8(void)
{
  const arm_status expected = ARM_MATH_SUCCESS;
  q7_t output[STRIDE2PAD1_DST_SIZE] = {0};

  const int32_t buf_size = arm_convolve_s8_get_buffer_size(STRIDE2PAD1_IN_CH,
                                                           STRIDE2PAD1_FILTER_X,
                                                           STRIDE2PAD1_FILTER_Y);
  q15_t *bufferA = (q15_t*)malloc(buf_size);

  arm_status result = arm_convolve_s8(stride2pad1_input,
                                      STRIDE2PAD1_INPUT_W,
                                      STRIDE2PAD1_INPUT_H,
                                      STRIDE2PAD1_IN_CH,
                                      STRIDE2PAD1_INPUT_BATCHES,
                                      stride2pad1_weights,
                                      STRIDE2PAD1_OUT_CH,
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
                                      STRIDE2PAD1_OUTPUT_OFFSET,
                                      STRIDE2PAD1_INPUT_OFFSET,
                                      STRIDE2PAD1_OUT_ACTIVATION_MIN,
                                      STRIDE2PAD1_OUT_ACTIVATION_MAX,
                                      STRIDE2PAD1_OUTPUT_W,
                                      STRIDE2PAD1_OUTPUT_H,
                                      bufferA);

  TEST_ASSERT_EQUAL(expected, result);
  TEST_ASSERT_TRUE(validate(output, stride2pad1_output_ref, STRIDE2PAD1_DST_SIZE));

  free(bufferA);
}

void conv_2_arm_convolve_s8(void)
{
  arm_status expected = ARM_MATH_SUCCESS;
  q7_t output[CONV_2_DST_SIZE] = {0};

  const int32_t buf_size = arm_convolve_s8_get_buffer_size(CONV_2_IN_CH,
                                                           CONV_2_FILTER_X,
                                                           CONV_2_FILTER_Y);
  q15_t *bufferA = (q15_t*)malloc(buf_size);

  arm_status result = arm_convolve_s8(conv_2_input,
                                      CONV_2_INPUT_W,
                                      CONV_2_INPUT_H,
                                      CONV_2_IN_CH,
                                      CONV_2_INPUT_BATCHES,
                                      conv_2_weights,
                                      CONV_2_OUT_CH,
                                      CONV_2_FILTER_X,
                                      CONV_2_FILTER_Y,
                                      CONV_2_PAD_X,
                                      CONV_2_PAD_Y,
                                      CONV_2_STRIDE_X,
                                      CONV_2_STRIDE_Y,
                                      conv_2_biases,
                                      output,
                                      conv_2_output_shift,
                                      conv_2_output_mult,
                                      CONV_2_OUTPUT_OFFSET,
                                      CONV_2_INPUT_OFFSET,
                                      CONV_2_OUT_ACTIVATION_MIN,
                                      CONV_2_OUT_ACTIVATION_MAX,
                                      CONV_2_OUTPUT_W,
                                      CONV_2_OUTPUT_H,
                                      bufferA);

  TEST_ASSERT_EQUAL(expected, result);
  TEST_ASSERT_TRUE(validate(output, conv_2_output_ref, CONV_2_DST_SIZE));

  free(bufferA);
}

void conv_3_arm_convolve_s8(void)
{
  arm_status expected = ARM_MATH_SUCCESS;
  q7_t output[CONV_3_DST_SIZE] = {0};

  const int32_t buf_size = arm_convolve_s8_get_buffer_size(CONV_3_IN_CH,
                                                           CONV_3_FILTER_X,
                                                           CONV_3_FILTER_Y);
  q15_t *bufferA = (q15_t*)malloc(buf_size);

  arm_status result = arm_convolve_s8(conv_3_input,
                                      CONV_3_INPUT_W,
                                      CONV_3_INPUT_H,
                                      CONV_3_IN_CH,
                                      CONV_3_INPUT_BATCHES,
                                      conv_3_weights,
                                      CONV_3_OUT_CH,
                                      CONV_3_FILTER_X,
                                      CONV_3_FILTER_Y,
                                      CONV_3_PAD_X,
                                      CONV_3_PAD_Y,
                                      CONV_3_STRIDE_X,
                                      CONV_3_STRIDE_Y,
                                      conv_3_biases,
                                      output,
                                      conv_3_output_shift,
                                      conv_3_output_mult,
                                      CONV_3_OUTPUT_OFFSET,
                                      CONV_3_INPUT_OFFSET,
                                      CONV_3_OUT_ACTIVATION_MIN,
                                      CONV_3_OUT_ACTIVATION_MAX,
                                      CONV_3_OUTPUT_W,
                                      CONV_3_OUTPUT_H,
                                      bufferA);

  TEST_ASSERT_EQUAL(expected, result);
  TEST_ASSERT_TRUE(validate(output, conv_3_output_ref, CONV_3_DST_SIZE));

  free(bufferA);
}

void conv_1_x_n_1_arm_convolve_s8(void)
{
  arm_status expected = ARM_MATH_SUCCESS;
  q7_t output[CONV_1_X_N_1_DST_SIZE] = {0};

  const int32_t buf_size = arm_convolve_s8_get_buffer_size(CONV_1_X_N_1_IN_CH,
                                                           CONV_1_X_N_1_FILTER_X,
                                                           CONV_1_X_N_1_FILTER_Y);
  q15_t *bufferA = (q15_t*)malloc(buf_size);

  arm_status result = arm_convolve_s8(conv_1_x_n_1_input,
                                      CONV_1_X_N_1_INPUT_W,
                                      CONV_1_X_N_1_INPUT_H,
                                      CONV_1_X_N_1_IN_CH,
                                      CONV_1_X_N_1_INPUT_BATCHES,
                                      conv_1_x_n_1_weights,
                                      CONV_1_X_N_1_OUT_CH,
                                      CONV_1_X_N_1_FILTER_X,
                                      CONV_1_X_N_1_FILTER_Y,
                                      CONV_1_X_N_1_PAD_X,
                                      CONV_1_X_N_1_PAD_Y,
                                      CONV_1_X_N_1_STRIDE_X,
                                      CONV_1_X_N_1_STRIDE_Y,
                                      conv_1_x_n_1_biases,
                                      output,
                                      conv_1_x_n_1_output_shift,
                                      conv_1_x_n_1_output_mult,
                                      CONV_1_X_N_1_OUTPUT_OFFSET,
                                      CONV_1_X_N_1_INPUT_OFFSET,
                                      CONV_1_X_N_1_OUT_ACTIVATION_MIN,
                                      CONV_1_X_N_1_OUT_ACTIVATION_MAX,
                                      CONV_1_X_N_1_OUTPUT_W,
                                      CONV_1_X_N_1_OUTPUT_H,
                                      bufferA);

  TEST_ASSERT_EQUAL(expected, result);
  TEST_ASSERT_TRUE(validate(output, conv_1_x_n_1_output_ref, CONV_1_X_N_1_DST_SIZE));

  free(bufferA);
}

void conv_1_x_n_1_1_arm_convolve_s8(void)
{
  arm_status expected = ARM_MATH_SIZE_MISMATCH;
  q7_t output[CONV_1_X_N_1_DST_SIZE] = {0};

  const int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(CONV_1_X_N_1_IN_CH,
                                                                 CONV_1_X_N_1_FILTER_X,
                                                                 CONV_1_X_N_1_FILTER_Y);
  q15_t *bufferA = (q15_t*)malloc(buf_size);

  arm_status result = arm_convolve_1_x_n_s8(conv_1_x_n_1_input,
                                            CONV_1_X_N_1_INPUT_W,
                                            CONV_1_X_N_1_IN_CH,
                                            CONV_1_X_N_1_INPUT_BATCHES,
                                            conv_1_x_n_1_weights,
                                            CONV_1_X_N_1_OUT_CH,
                                            CONV_1_X_N_1_FILTER_X,
                                            CONV_1_X_N_1_PAD_X,
                                            CONV_1_X_N_1_STRIDE_X,
                                            conv_1_x_n_1_biases,
                                            output,
                                            conv_1_x_n_1_output_shift,
                                            conv_1_x_n_1_output_mult,
                                            CONV_1_X_N_1_OUTPUT_OFFSET,
                                            CONV_1_X_N_1_INPUT_OFFSET,
                                            CONV_1_X_N_1_OUT_ACTIVATION_MIN,
                                            CONV_1_X_N_1_OUT_ACTIVATION_MAX,
                                            CONV_1_X_N_1_OUTPUT_W,
                                            bufferA);
  free(bufferA);
  TEST_ASSERT_EQUAL(expected, result);
}

void conv_1_x_n_2_arm_convolve_s8(void)
{
  arm_status expected = ARM_MATH_SIZE_MISMATCH;
  q7_t output[CONV_1_X_N_2_DST_SIZE] = {0};

  const int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(CONV_1_X_N_2_IN_CH,
                                                                 CONV_1_X_N_2_FILTER_X,
                                                                 CONV_1_X_N_2_FILTER_Y);
  q15_t *bufferA = (q15_t*)malloc(buf_size);

  arm_status result = arm_convolve_1_x_n_s8(conv_1_x_n_2_input,
                                            CONV_1_X_N_2_INPUT_W,
                                            CONV_1_X_N_2_IN_CH,
                                            CONV_1_X_N_2_INPUT_BATCHES,
                                            conv_1_x_n_2_weights,
                                            CONV_1_X_N_2_OUT_CH,
                                            CONV_1_X_N_2_FILTER_X,
                                            CONV_1_X_N_2_PAD_X,
                                            CONV_1_X_N_2_STRIDE_X,
                                            conv_1_x_n_2_biases,
                                            output,
                                            conv_1_x_n_2_output_shift,
                                            conv_1_x_n_2_output_mult,
                                            CONV_1_X_N_2_OUTPUT_OFFSET,
                                            CONV_1_X_N_2_INPUT_OFFSET,
                                            CONV_1_X_N_2_OUT_ACTIVATION_MIN,
                                            CONV_1_X_N_2_OUT_ACTIVATION_MAX,
                                            CONV_1_X_N_2_OUTPUT_W,
                                            bufferA);
  free(bufferA);
  TEST_ASSERT_EQUAL(expected, result);
}

void conv_1_x_n_3_arm_convolve_s8(void)
{
  arm_status expected = ARM_MATH_SIZE_MISMATCH;
  q7_t output[CONV_1_X_N_3_DST_SIZE] = {0};

  const int32_t buf_size = arm_convolve_1_x_n_s8_get_buffer_size(CONV_1_X_N_3_IN_CH,
                                                                 CONV_1_X_N_3_FILTER_X,
                                                                 CONV_1_X_N_3_FILTER_Y);
  q15_t *bufferA = (q15_t*)malloc(buf_size);

  arm_status result = arm_convolve_1_x_n_s8(conv_1_x_n_3_input,
                                            CONV_1_X_N_3_INPUT_W,
                                            CONV_1_X_N_3_IN_CH,
                                            CONV_1_X_N_3_INPUT_BATCHES,
                                            conv_1_x_n_3_weights,
                                            CONV_1_X_N_3_OUT_CH,
                                            CONV_1_X_N_3_FILTER_X,
                                            CONV_1_X_N_3_PAD_X,
                                            CONV_1_X_N_3_STRIDE_X,
                                            conv_1_x_n_3_biases,
                                            output,
                                            conv_1_x_n_3_output_shift,
                                            conv_1_x_n_3_output_mult,
                                            CONV_1_X_N_3_OUTPUT_OFFSET,
                                            CONV_1_X_N_3_INPUT_OFFSET,
                                            CONV_1_X_N_3_OUT_ACTIVATION_MIN,
                                            CONV_1_X_N_3_OUT_ACTIVATION_MAX,
                                            CONV_1_X_N_3_OUTPUT_W,
                                            bufferA);
  free(bufferA);
  TEST_ASSERT_EQUAL(expected, result);
}