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
#include <unity.h>

#include "../TestData/fully_connected/test_data.h"
#include "../TestData/fully_connected_mve_0/test_data.h"
#include "../TestData/fully_connected_mve_1/test_data.h"
#include "../TestData/fully_connected_null_bias_0/test_data.h"
#include "../TestData/fully_connected_out_activation/test_data.h"
#include "../Utils/validate.h"

void fully_connected_arm_fully_connected_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[FULLY_CONNECTED_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = fully_connected_biases;
    const q7_t *kernel_data = fully_connected_weights;
    const q7_t *input_data = fully_connected_input;
    const q7_t *output_ref = fully_connected_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_DST_SIZE;

    input_dims.n = FULLY_CONNECTED_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_INPUT_W;
    input_dims.h = FULLY_CONNECTED_INPUT_H;
    input_dims.c = FULLY_CONNECTED_IN_CH;
    filter_dims.n = FULLY_CONNECTED_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_OUT_CH;
    output_dims.n = FULLY_CONNECTED_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_OUT_CH;

    fc_params.input_offset = FULLY_CONNECTED_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_OUT_ACTIVATION_MAX;

    quant_params.multiplier = FULLY_CONNECTED_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_OUTPUT_SHIFT;

    int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    arm_status result = arm_fully_connected_s8(&ctx,
                                               &fc_params,
                                               &quant_params,
                                               &input_dims,
                                               input_data,
                                               &filter_dims,
                                               kernel_data,
                                               &bias_dims,
                                               bias_data,
                                               &output_dims,
                                               output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void fully_connected_mve_0_arm_fully_connected_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[FULLY_CONNECTED_MVE_0_DST_SIZE] = {0};
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    const q31_t *bias_data = fully_connected_mve_0_biases;
    const q7_t *kernel_data = fully_connected_mve_0_weights;
    const q7_t *input_data = fully_connected_mve_0_input;
    const q7_t *output_ref = fully_connected_mve_0_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_MVE_0_DST_SIZE;
    input_dims.n = FULLY_CONNECTED_MVE_0_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_MVE_0_INPUT_W;
    input_dims.h = FULLY_CONNECTED_MVE_0_INPUT_H;
    input_dims.c = FULLY_CONNECTED_MVE_0_IN_CH;
    filter_dims.n = FULLY_CONNECTED_MVE_0_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_MVE_0_OUT_CH;
    output_dims.n = FULLY_CONNECTED_MVE_0_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_MVE_0_OUT_CH;
    fc_params.input_offset = FULLY_CONNECTED_MVE_0_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_MVE_0_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_MVE_0_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_MVE_0_OUT_ACTIVATION_MAX;
    quant_params.multiplier = FULLY_CONNECTED_MVE_0_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_MVE_0_OUTPUT_SHIFT;

    int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;
    arm_status result = arm_fully_connected_s8(&ctx,
                                               &fc_params,
                                               &quant_params,
                                               &input_dims,
                                               input_data,
                                               &filter_dims,
                                               kernel_data,
                                               &bias_dims,
                                               bias_data,
                                               &output_dims,
                                               output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void fully_connected_mve_1_arm_fully_connected_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[FULLY_CONNECTED_MVE_1_DST_SIZE] = {0};
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    const q31_t *bias_data = fully_connected_mve_1_biases;
    const q7_t *kernel_data = fully_connected_mve_1_weights;
    const q7_t *input_data = fully_connected_mve_1_input;
    const q7_t *output_ref = fully_connected_mve_1_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_MVE_1_DST_SIZE;
    input_dims.n = FULLY_CONNECTED_MVE_1_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_MVE_1_INPUT_W;
    input_dims.h = FULLY_CONNECTED_MVE_1_INPUT_H;
    input_dims.c = FULLY_CONNECTED_MVE_1_IN_CH;
    filter_dims.n = FULLY_CONNECTED_MVE_1_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_MVE_1_OUT_CH;
    output_dims.n = FULLY_CONNECTED_MVE_1_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_MVE_1_OUT_CH;
    fc_params.input_offset = FULLY_CONNECTED_MVE_1_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_MVE_1_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_MVE_1_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_MVE_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = FULLY_CONNECTED_MVE_1_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_MVE_1_OUTPUT_SHIFT;

    int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;
    arm_status result = arm_fully_connected_s8(&ctx,
                                               &fc_params,
                                               &quant_params,
                                               &input_dims,
                                               input_data,
                                               &filter_dims,
                                               kernel_data,
                                               &bias_dims,
                                               bias_data,
                                               &output_dims,
                                               output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void fully_connected_null_bias_0_arm_fully_connected_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[FULLY_CONNECTED_NULL_BIAS_0_DST_SIZE] = {0};
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    const q31_t *bias_data = fully_connected_null_bias_0_biases;
    const q7_t *kernel_data = fully_connected_null_bias_0_weights;
    const q7_t *input_data = fully_connected_null_bias_0_input;
    const q7_t *output_ref = fully_connected_null_bias_0_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_NULL_BIAS_0_DST_SIZE;
    input_dims.n = FULLY_CONNECTED_NULL_BIAS_0_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_NULL_BIAS_0_INPUT_W;
    input_dims.h = FULLY_CONNECTED_NULL_BIAS_0_INPUT_H;
    input_dims.c = FULLY_CONNECTED_NULL_BIAS_0_IN_CH;
    filter_dims.n = FULLY_CONNECTED_NULL_BIAS_0_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_NULL_BIAS_0_OUT_CH;
    output_dims.n = FULLY_CONNECTED_NULL_BIAS_0_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_NULL_BIAS_0_OUT_CH;
    fc_params.input_offset = FULLY_CONNECTED_NULL_BIAS_0_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_NULL_BIAS_0_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_NULL_BIAS_0_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_NULL_BIAS_0_OUT_ACTIVATION_MAX;
    quant_params.multiplier = FULLY_CONNECTED_NULL_BIAS_0_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_NULL_BIAS_0_OUTPUT_SHIFT;

    arm_status ip_check = ARM_MATH_SUCCESS;
    for (int i = 0; i < FULLY_CONNECTED_NULL_BIAS_0_OUT_CH; i++)
    {
        if (bias_data[i] != 0)
        {
            ip_check = ARM_MATH_ARGUMENT_ERROR;
            break;
        }
    }
    TEST_ASSERT_EQUAL(expected, ip_check);

    int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;
    arm_status result = arm_fully_connected_s8(&ctx,
                                               &fc_params,
                                               &quant_params,
                                               &input_dims,
                                               input_data,
                                               &filter_dims,
                                               kernel_data,
                                               &bias_dims,
                                               NULL,
                                               &output_dims,
                                               output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}

void fully_connected_out_activation_arm_fully_connected_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[FULLY_CONNECTED_OUT_ACTIVATION_DST_SIZE] = {0};
    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;
    const q31_t *bias_data = fully_connected_out_activation_biases;
    const q7_t *kernel_data = fully_connected_out_activation_weights;
    const q7_t *input_data = fully_connected_out_activation_input;
    const q7_t *output_ref = fully_connected_out_activation_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_OUT_ACTIVATION_DST_SIZE;
    input_dims.n = FULLY_CONNECTED_OUT_ACTIVATION_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_OUT_ACTIVATION_INPUT_W;
    input_dims.h = FULLY_CONNECTED_OUT_ACTIVATION_INPUT_H;
    input_dims.c = FULLY_CONNECTED_OUT_ACTIVATION_IN_CH;
    filter_dims.n = FULLY_CONNECTED_OUT_ACTIVATION_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_OUT_ACTIVATION_OUT_CH;
    output_dims.n = FULLY_CONNECTED_OUT_ACTIVATION_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_OUT_ACTIVATION_OUT_CH;
    fc_params.input_offset = FULLY_CONNECTED_OUT_ACTIVATION_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_OUT_ACTIVATION_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_OUT_ACTIVATION_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_OUT_ACTIVATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = FULLY_CONNECTED_OUT_ACTIVATION_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_OUT_ACTIVATION_OUTPUT_SHIFT;

    int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;
    arm_status result = arm_fully_connected_s8(&ctx,
                                               &fc_params,
                                               &quant_params,
                                               &input_dims,
                                               input_data,
                                               &filter_dims,
                                               kernel_data,
                                               &bias_dims,
                                               bias_data,
                                               &output_dims,
                                               output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
}
