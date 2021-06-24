/*
 * Copyright (C) 2010-2021 Arm Limited or its affiliates. All rights reserved.
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

#include "../TestData/avgpooling/test_data.h"
#include "../TestData/avgpooling_1/test_data.h"
#include "../TestData/avgpooling_2/test_data.h"
#include "../TestData/avgpooling_3/test_data.h"
#include "../TestData/avgpooling_4/test_data.h"
#include "../TestData/avgpooling_5/test_data.h"
#include "../Utils/validate.h"

void avgpooling_arm_avgpool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[AVGPOOLING_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = avgpooling_input;

    input_dims.n = AVGPOOLING_INPUT_BATCHES;
    input_dims.w = AVGPOOLING_INPUT_W;
    input_dims.h = AVGPOOLING_INPUT_H;
    input_dims.c = AVGPOOLING_IN_CH;
    filter_dims.w = AVGPOOLING_FILTER_X;
    filter_dims.h = AVGPOOLING_FILTER_Y;
    output_dims.w = AVGPOOLING_OUTPUT_W;
    output_dims.h = AVGPOOLING_OUTPUT_H;
    output_dims.c = AVGPOOLING_OUT_CH;

    pool_params.padding.w = AVGPOOLING_PAD_X;
    pool_params.padding.h = AVGPOOLING_PAD_Y;
    pool_params.stride.w = AVGPOOLING_STRIDE_X;
    pool_params.stride.h = AVGPOOLING_STRIDE_Y;

    pool_params.activation.min = AVGPOOLING_OUT_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_OUT_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s8_get_buffer_size(AVGPOOLING_OUTPUT_W, AVGPOOLING_IN_CH);
    ctx.buf = malloc(ctx.size);

    arm_status result = arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, avgpooling_output_ref, AVGPOOLING_DST_SIZE));
}

void avgpooling_1_arm_avgpool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[AVGPOOLING_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = avgpooling_1_input;

    input_dims.n = AVGPOOLING_1_INPUT_BATCHES;
    input_dims.w = AVGPOOLING_1_INPUT_W;
    input_dims.h = AVGPOOLING_1_INPUT_H;
    input_dims.c = AVGPOOLING_1_IN_CH;
    filter_dims.w = AVGPOOLING_1_FILTER_X;
    filter_dims.h = AVGPOOLING_1_FILTER_Y;
    output_dims.w = AVGPOOLING_1_OUTPUT_W;
    output_dims.h = AVGPOOLING_1_OUTPUT_H;
    output_dims.c = AVGPOOLING_1_OUT_CH;

    pool_params.padding.w = AVGPOOLING_1_PAD_X;
    pool_params.padding.h = AVGPOOLING_1_PAD_Y;
    pool_params.stride.w = AVGPOOLING_1_STRIDE_X;
    pool_params.stride.h = AVGPOOLING_1_STRIDE_Y;

    pool_params.activation.min = AVGPOOLING_1_OUT_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_1_OUT_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s8_get_buffer_size(AVGPOOLING_1_OUTPUT_W, AVGPOOLING_1_IN_CH);
    ctx.buf = malloc(ctx.size);

    arm_status result = arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, avgpooling_1_output_ref, AVGPOOLING_1_DST_SIZE));
}

void avgpooling_2_arm_avgpool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[AVGPOOLING_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = avgpooling_2_input;

    input_dims.n = AVGPOOLING_2_INPUT_BATCHES;
    input_dims.w = AVGPOOLING_2_INPUT_W;
    input_dims.h = AVGPOOLING_2_INPUT_H;
    input_dims.c = AVGPOOLING_2_IN_CH;
    filter_dims.w = AVGPOOLING_2_FILTER_X;
    filter_dims.h = AVGPOOLING_2_FILTER_Y;
    output_dims.w = AVGPOOLING_2_OUTPUT_W;
    output_dims.h = AVGPOOLING_2_OUTPUT_H;
    output_dims.c = AVGPOOLING_2_OUT_CH;

    pool_params.padding.w = AVGPOOLING_2_PAD_X;
    pool_params.padding.h = AVGPOOLING_2_PAD_Y;
    pool_params.stride.w = AVGPOOLING_2_STRIDE_X;
    pool_params.stride.h = AVGPOOLING_2_STRIDE_Y;

    pool_params.activation.min = AVGPOOLING_2_OUT_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_2_OUT_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s8_get_buffer_size(AVGPOOLING_2_OUTPUT_W, AVGPOOLING_2_IN_CH);
    ctx.buf = malloc(ctx.size);

    arm_status result = arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, avgpooling_2_output_ref, AVGPOOLING_2_DST_SIZE));
}

void avgpooling_3_arm_avgpool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[AVGPOOLING_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = avgpooling_3_input;

    input_dims.n = AVGPOOLING_3_INPUT_BATCHES;
    input_dims.w = AVGPOOLING_3_INPUT_W;
    input_dims.h = AVGPOOLING_3_INPUT_H;
    input_dims.c = AVGPOOLING_3_IN_CH;
    filter_dims.w = AVGPOOLING_3_FILTER_X;
    filter_dims.h = AVGPOOLING_3_FILTER_Y;
    output_dims.w = AVGPOOLING_3_OUTPUT_W;
    output_dims.h = AVGPOOLING_3_OUTPUT_H;
    output_dims.c = AVGPOOLING_3_OUT_CH;

    pool_params.padding.w = AVGPOOLING_3_PAD_X;
    pool_params.padding.h = AVGPOOLING_3_PAD_Y;
    pool_params.stride.w = AVGPOOLING_3_STRIDE_X;
    pool_params.stride.h = AVGPOOLING_3_STRIDE_Y;

    pool_params.activation.min = AVGPOOLING_3_OUT_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_3_OUT_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s8_get_buffer_size(AVGPOOLING_3_OUTPUT_W, AVGPOOLING_3_IN_CH);
    ctx.buf = malloc(ctx.size);

    arm_status result = arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, avgpooling_3_output_ref, AVGPOOLING_3_DST_SIZE));
}

void avgpooling_4_arm_avgpool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[AVGPOOLING_4_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = avgpooling_4_input;

    input_dims.n = AVGPOOLING_4_INPUT_BATCHES;
    input_dims.w = AVGPOOLING_4_INPUT_W;
    input_dims.h = AVGPOOLING_4_INPUT_H;
    input_dims.c = AVGPOOLING_4_IN_CH;
    filter_dims.w = AVGPOOLING_4_FILTER_X;
    filter_dims.h = AVGPOOLING_4_FILTER_Y;
    output_dims.w = AVGPOOLING_4_OUTPUT_W;
    output_dims.h = AVGPOOLING_4_OUTPUT_H;
    output_dims.c = AVGPOOLING_4_OUT_CH;

    pool_params.padding.w = AVGPOOLING_4_PAD_X;
    pool_params.padding.h = AVGPOOLING_4_PAD_Y;
    pool_params.stride.w = AVGPOOLING_4_STRIDE_X;
    pool_params.stride.h = AVGPOOLING_4_STRIDE_Y;

    pool_params.activation.min = AVGPOOLING_4_OUT_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_4_OUT_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s8_get_buffer_size(AVGPOOLING_4_OUTPUT_W, AVGPOOLING_4_IN_CH);
    ctx.buf = malloc(ctx.size);

    arm_status result = arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, avgpooling_4_output_ref, AVGPOOLING_4_DST_SIZE));
}

void avgpooling_5_arm_avgpool_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[AVGPOOLING_5_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const q7_t *input_data = avgpooling_5_input;

    input_dims.n = AVGPOOLING_5_INPUT_BATCHES;
    input_dims.w = AVGPOOLING_5_INPUT_W;
    input_dims.h = AVGPOOLING_5_INPUT_H;
    input_dims.c = AVGPOOLING_5_IN_CH;
    filter_dims.w = AVGPOOLING_5_FILTER_X;
    filter_dims.h = AVGPOOLING_5_FILTER_Y;
    output_dims.w = AVGPOOLING_5_OUTPUT_W;
    output_dims.h = AVGPOOLING_5_OUTPUT_H;
    output_dims.c = AVGPOOLING_5_OUT_CH;

    pool_params.padding.w = AVGPOOLING_5_PAD_X;
    pool_params.padding.h = AVGPOOLING_5_PAD_Y;
    pool_params.stride.w = AVGPOOLING_5_STRIDE_X;
    pool_params.stride.h = AVGPOOLING_5_STRIDE_Y;

    pool_params.activation.min = AVGPOOLING_5_OUT_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_5_OUT_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s8_get_buffer_size(AVGPOOLING_5_OUTPUT_W, AVGPOOLING_5_IN_CH);
    ctx.buf = malloc(ctx.size);

    arm_status result = arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, avgpooling_5_output_ref, AVGPOOLING_5_DST_SIZE));
}
