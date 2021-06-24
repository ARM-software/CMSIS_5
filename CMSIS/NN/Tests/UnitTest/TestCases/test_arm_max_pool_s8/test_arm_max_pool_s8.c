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

#include "unity.h"
#include <arm_nnfunctions.h>

#include "../TestData/maxpooling/test_data.h"
#include "../TestData/maxpooling_1/test_data.h"
#include "../TestData/maxpooling_2/test_data.h"
#include "../TestData/maxpooling_3/test_data.h"
#include "../TestData/maxpooling_4/test_data.h"
#include "../TestData/maxpooling_5/test_data.h"
#include "../TestData/maxpooling_6/test_data.h"
#include "../TestData/maxpooling_7/test_data.h"
#include "../Utils/validate.h"

#define REPEAT_NUM (2)

void maxpooling_arm_max_pool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[MAXPOOLING_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = maxpooling_input;

    input_dims.n = MAXPOOLING_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_INPUT_W;
    input_dims.h = MAXPOOLING_INPUT_H;
    input_dims.c = MAXPOOLING_IN_CH;
    filter_dims.w = MAXPOOLING_FILTER_X;
    filter_dims.h = MAXPOOLING_FILTER_Y;
    output_dims.w = MAXPOOLING_OUTPUT_W;
    output_dims.h = MAXPOOLING_OUTPUT_H;
    output_dims.c = MAXPOOLING_OUT_CH;

    pool_params.padding.w = MAXPOOLING_PAD_X;
    pool_params.padding.h = MAXPOOLING_PAD_Y;
    pool_params.stride.w = MAXPOOLING_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_OUT_ACTIVATION_MAX;

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_status result =
            arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        TEST_ASSERT_EQUAL(expected, result);
        TEST_ASSERT_TRUE(validate(output, maxpooling_output_ref, MAXPOOLING_DST_SIZE));
    }
}

void maxpooling_1_arm_max_pool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[MAXPOOLING_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = maxpooling_1_input;

    input_dims.n = MAXPOOLING_1_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_1_INPUT_W;
    input_dims.h = MAXPOOLING_1_INPUT_H;
    input_dims.c = MAXPOOLING_1_IN_CH;
    filter_dims.w = MAXPOOLING_1_FILTER_X;
    filter_dims.h = MAXPOOLING_1_FILTER_Y;
    output_dims.w = MAXPOOLING_1_OUTPUT_W;
    output_dims.h = MAXPOOLING_1_OUTPUT_H;
    output_dims.c = MAXPOOLING_1_OUT_CH;

    pool_params.padding.w = MAXPOOLING_1_PAD_X;
    pool_params.padding.h = MAXPOOLING_1_PAD_Y;
    pool_params.stride.w = MAXPOOLING_1_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_1_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_1_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_1_OUT_ACTIVATION_MAX;

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_status result =
            arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        TEST_ASSERT_EQUAL(expected, result);
        TEST_ASSERT_TRUE(validate(output, maxpooling_1_output_ref, MAXPOOLING_1_DST_SIZE));
    }
}

void maxpooling_2_arm_max_pool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[MAXPOOLING_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = maxpooling_2_input;

    input_dims.n = MAXPOOLING_2_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_2_INPUT_W;
    input_dims.h = MAXPOOLING_2_INPUT_H;
    input_dims.c = MAXPOOLING_2_IN_CH;
    filter_dims.w = MAXPOOLING_2_FILTER_X;
    filter_dims.h = MAXPOOLING_2_FILTER_Y;
    output_dims.w = MAXPOOLING_2_OUTPUT_W;
    output_dims.h = MAXPOOLING_2_OUTPUT_H;
    output_dims.c = MAXPOOLING_2_OUT_CH;

    pool_params.padding.w = MAXPOOLING_2_PAD_X;
    pool_params.padding.h = MAXPOOLING_2_PAD_Y;
    pool_params.stride.w = MAXPOOLING_2_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_2_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_2_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_2_OUT_ACTIVATION_MAX;

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_status result =
            arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        TEST_ASSERT_EQUAL(expected, result);
        TEST_ASSERT_TRUE(validate(output, maxpooling_2_output_ref, MAXPOOLING_2_DST_SIZE));
    }
}

void maxpooling_3_arm_max_pool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[MAXPOOLING_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = maxpooling_3_input;

    input_dims.n = MAXPOOLING_3_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_3_INPUT_W;
    input_dims.h = MAXPOOLING_3_INPUT_H;
    input_dims.c = MAXPOOLING_3_IN_CH;
    filter_dims.w = MAXPOOLING_3_FILTER_X;
    filter_dims.h = MAXPOOLING_3_FILTER_Y;
    output_dims.w = MAXPOOLING_3_OUTPUT_W;
    output_dims.h = MAXPOOLING_3_OUTPUT_H;
    output_dims.c = MAXPOOLING_3_OUT_CH;

    pool_params.padding.w = MAXPOOLING_3_PAD_X;
    pool_params.padding.h = MAXPOOLING_3_PAD_Y;
    pool_params.stride.w = MAXPOOLING_3_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_3_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_3_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_3_OUT_ACTIVATION_MAX;

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_status result =
            arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        TEST_ASSERT_EQUAL(expected, result);
        TEST_ASSERT_TRUE(validate(output, maxpooling_3_output_ref, MAXPOOLING_3_DST_SIZE));
    }
}

void maxpooling_4_arm_max_pool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[MAXPOOLING_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = maxpooling_4_input;

    input_dims.n = MAXPOOLING_4_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_4_INPUT_W;
    input_dims.h = MAXPOOLING_4_INPUT_H;
    input_dims.c = MAXPOOLING_4_IN_CH;
    filter_dims.w = MAXPOOLING_4_FILTER_X;
    filter_dims.h = MAXPOOLING_4_FILTER_Y;
    output_dims.w = MAXPOOLING_4_OUTPUT_W;
    output_dims.h = MAXPOOLING_4_OUTPUT_H;
    output_dims.c = MAXPOOLING_4_OUT_CH;

    pool_params.padding.w = MAXPOOLING_4_PAD_X;
    pool_params.padding.h = MAXPOOLING_4_PAD_Y;
    pool_params.stride.w = MAXPOOLING_4_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_4_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_4_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_4_OUT_ACTIVATION_MAX;

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_status result =
            arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        TEST_ASSERT_EQUAL(expected, result);
        TEST_ASSERT_TRUE(validate(output, maxpooling_4_output_ref, MAXPOOLING_4_DST_SIZE));
    }
}

void maxpooling_5_arm_max_pool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[MAXPOOLING_5_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = maxpooling_5_input;

    input_dims.n = MAXPOOLING_5_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_5_INPUT_W;
    input_dims.h = MAXPOOLING_5_INPUT_H;
    input_dims.c = MAXPOOLING_5_IN_CH;
    filter_dims.w = MAXPOOLING_5_FILTER_X;
    filter_dims.h = MAXPOOLING_5_FILTER_Y;
    output_dims.w = MAXPOOLING_5_OUTPUT_W;
    output_dims.h = MAXPOOLING_5_OUTPUT_H;
    output_dims.c = MAXPOOLING_5_OUT_CH;

    pool_params.padding.w = MAXPOOLING_5_PAD_X;
    pool_params.padding.h = MAXPOOLING_5_PAD_Y;
    pool_params.stride.w = MAXPOOLING_5_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_5_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_5_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_5_OUT_ACTIVATION_MAX;

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_status result =
            arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        TEST_ASSERT_EQUAL(expected, result);
        TEST_ASSERT_TRUE(validate(output, maxpooling_5_output_ref, MAXPOOLING_5_DST_SIZE));
    }
}

void maxpooling_6_arm_max_pool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[MAXPOOLING_6_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = maxpooling_6_input;

    input_dims.n = MAXPOOLING_6_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_6_INPUT_W;
    input_dims.h = MAXPOOLING_6_INPUT_H;
    input_dims.c = MAXPOOLING_6_IN_CH;
    filter_dims.w = MAXPOOLING_6_FILTER_X;
    filter_dims.h = MAXPOOLING_6_FILTER_Y;
    output_dims.w = MAXPOOLING_6_OUTPUT_W;
    output_dims.h = MAXPOOLING_6_OUTPUT_H;
    output_dims.c = MAXPOOLING_6_OUT_CH;

    pool_params.padding.w = MAXPOOLING_6_PAD_X;
    pool_params.padding.h = MAXPOOLING_6_PAD_Y;
    pool_params.stride.w = MAXPOOLING_6_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_6_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_6_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_6_OUT_ACTIVATION_MAX;

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_status result =
            arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        TEST_ASSERT_EQUAL(expected, result);
        TEST_ASSERT_TRUE(validate(output, maxpooling_6_output_ref, MAXPOOLING_6_DST_SIZE));
    }
}

void maxpooling_7_arm_max_pool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[MAXPOOLING_7_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = maxpooling_7_input;

    input_dims.n = MAXPOOLING_7_INPUT_BATCHES;
    input_dims.w = MAXPOOLING_7_INPUT_W;
    input_dims.h = MAXPOOLING_7_INPUT_H;
    input_dims.c = MAXPOOLING_7_IN_CH;
    filter_dims.w = MAXPOOLING_7_FILTER_X;
    filter_dims.h = MAXPOOLING_7_FILTER_Y;
    output_dims.w = MAXPOOLING_7_OUTPUT_W;
    output_dims.h = MAXPOOLING_7_OUTPUT_H;
    output_dims.c = MAXPOOLING_7_OUT_CH;

    pool_params.padding.w = MAXPOOLING_7_PAD_X;
    pool_params.padding.h = MAXPOOLING_7_PAD_Y;
    pool_params.stride.w = MAXPOOLING_7_STRIDE_X;
    pool_params.stride.h = MAXPOOLING_7_STRIDE_Y;

    pool_params.activation.min = MAXPOOLING_7_OUT_ACTIVATION_MIN;
    pool_params.activation.max = MAXPOOLING_7_OUT_ACTIVATION_MAX;

    for (int i = 0; i < REPEAT_NUM; i++)
    {
        arm_status result =
            arm_max_pool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

        TEST_ASSERT_EQUAL(expected, result);
        TEST_ASSERT_TRUE(validate(output, maxpooling_7_output_ref, MAXPOOLING_7_DST_SIZE));
    }
}
