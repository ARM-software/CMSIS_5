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
#include <unity.h>

#include "../TestData/basic/test_data.h"
#include "../TestData/depthwise_2/test_data.h"
#include "../TestData/depthwise_mult_batches/test_data.h"
#include "../TestData/depthwise_out_activation/test_data.h"
#include "../TestData/stride2pad1/test_data.h"
#include "../Utils/validate.h"

void basic_arm_depthwise_conv_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[BASIC_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = basic_biases;
    const q7_t *input_data = basic_input;

    input_dims.n = BASIC_INPUT_BATCHES;
    input_dims.w = BASIC_INPUT_W;
    input_dims.h = BASIC_INPUT_H;
    input_dims.c = BASIC_IN_CH;
    filter_dims.w = BASIC_FILTER_X;
    filter_dims.h = BASIC_FILTER_Y;
    output_dims.w = BASIC_OUTPUT_W;
    output_dims.h = BASIC_OUTPUT_H;
    output_dims.c = BASIC_OUT_CH;

    dw_conv_params.padding.w = BASIC_PAD_X;
    dw_conv_params.padding.h = BASIC_PAD_Y;
    dw_conv_params.stride.w = BASIC_STRIDE_X;
    dw_conv_params.stride.h = BASIC_STRIDE_Y;
    dw_conv_params.ch_mult = 1;

    dw_conv_params.input_offset = BASIC_INPUT_OFFSET;
    dw_conv_params.output_offset = BASIC_OUTPUT_OFFSET;
    dw_conv_params.activation.min = BASIC_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = BASIC_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)basic_output_mult;
    quant_params.shift = (int32_t *)basic_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_status result = arm_depthwise_conv_s8(&ctx,
                                              &dw_conv_params,
                                              &quant_params,
                                              &input_dims,
                                              input_data,
                                              &filter_dims,
                                              basic_weights,
                                              &bias_dims,
                                              bias_data,
                                              &output_dims,
                                              output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, basic_output_ref, BASIC_DST_SIZE));
}

void stride2pad1_arm_depthwise_conv_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[STRIDE2PAD1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = stride2pad1_biases;
    const q7_t *kernel_data = stride2pad1_weights;
    const q7_t *input_data = stride2pad1_input;

    input_dims.n = STRIDE2PAD1_INPUT_BATCHES;
    input_dims.w = STRIDE2PAD1_INPUT_W;
    input_dims.h = STRIDE2PAD1_INPUT_H;
    input_dims.c = STRIDE2PAD1_IN_CH;
    filter_dims.w = STRIDE2PAD1_FILTER_X;
    filter_dims.h = STRIDE2PAD1_FILTER_Y;
    output_dims.w = STRIDE2PAD1_OUTPUT_W;
    output_dims.h = STRIDE2PAD1_OUTPUT_H;
    output_dims.c = STRIDE2PAD1_OUT_CH;

    dw_conv_params.padding.w = STRIDE2PAD1_PAD_X;
    dw_conv_params.padding.h = STRIDE2PAD1_PAD_Y;
    dw_conv_params.stride.w = STRIDE2PAD1_STRIDE_X;
    dw_conv_params.stride.h = STRIDE2PAD1_STRIDE_Y;
    dw_conv_params.ch_mult = 1;

    dw_conv_params.input_offset = STRIDE2PAD1_INPUT_OFFSET;
    dw_conv_params.output_offset = STRIDE2PAD1_OUTPUT_OFFSET;
    dw_conv_params.activation.min = STRIDE2PAD1_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = STRIDE2PAD1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)stride2pad1_output_mult;
    quant_params.shift = (int32_t *)stride2pad1_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_status result = arm_depthwise_conv_s8(&ctx,
                                              &dw_conv_params,
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
    TEST_ASSERT_TRUE(validate(output, stride2pad1_output_ref, STRIDE2PAD1_DST_SIZE));
}

void depthwise_2_arm_depthwise_conv_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[DEPTHWISE_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = depthwise_2_biases;
    const q7_t *kernel_data = depthwise_2_weights;
    const q7_t *input_data = depthwise_2_input;

    input_dims.n = DEPTHWISE_2_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_2_INPUT_W;
    input_dims.h = DEPTHWISE_2_INPUT_H;
    input_dims.c = DEPTHWISE_2_IN_CH;
    filter_dims.w = DEPTHWISE_2_FILTER_X;
    filter_dims.h = DEPTHWISE_2_FILTER_Y;
    output_dims.w = DEPTHWISE_2_OUTPUT_W;
    output_dims.h = DEPTHWISE_2_OUTPUT_H;
    output_dims.c = DEPTHWISE_2_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_2_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_2_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_2_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_2_STRIDE_Y;
    dw_conv_params.ch_mult = DEPTHWISE_2_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_2_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_2_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_2_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_2_output_mult;
    quant_params.shift = (int32_t *)depthwise_2_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_status result = arm_depthwise_conv_s8(&ctx,
                                              &dw_conv_params,
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
    TEST_ASSERT_TRUE(validate(output, depthwise_2_output_ref, DEPTHWISE_2_DST_SIZE));
}

void depthwise_out_activation_arm_depthwise_conv_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[DEPTHWISE_OUT_ACTIVATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = depthwise_out_activation_biases;
    const q7_t *kernel_data = depthwise_out_activation_weights;
    const q7_t *input_data = depthwise_out_activation_input;

    input_dims.n = DEPTHWISE_OUT_ACTIVATION_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_OUT_ACTIVATION_INPUT_W;
    input_dims.h = DEPTHWISE_OUT_ACTIVATION_INPUT_H;
    input_dims.c = DEPTHWISE_OUT_ACTIVATION_IN_CH;
    filter_dims.w = DEPTHWISE_OUT_ACTIVATION_FILTER_X;
    filter_dims.h = DEPTHWISE_OUT_ACTIVATION_FILTER_Y;
    output_dims.w = DEPTHWISE_OUT_ACTIVATION_OUTPUT_W;
    output_dims.h = DEPTHWISE_OUT_ACTIVATION_OUTPUT_H;
    output_dims.c = DEPTHWISE_OUT_ACTIVATION_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_OUT_ACTIVATION_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_OUT_ACTIVATION_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_OUT_ACTIVATION_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_OUT_ACTIVATION_STRIDE_Y;
    dw_conv_params.ch_mult = DEPTHWISE_OUT_ACTIVATION_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_OUT_ACTIVATION_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_OUT_ACTIVATION_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_OUT_ACTIVATION_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_OUT_ACTIVATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_out_activation_output_mult;
    quant_params.shift = (int32_t *)depthwise_out_activation_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_status result = arm_depthwise_conv_s8(&ctx,
                                              &dw_conv_params,
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
    TEST_ASSERT_TRUE(validate(output, depthwise_out_activation_output_ref, DEPTHWISE_OUT_ACTIVATION_DST_SIZE));
}

void depthwise_mult_batches_arm_depthwise_conv_s8(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[DEPTHWISE_MULT_BATCHES_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = depthwise_mult_batches_biases;
    const q7_t *kernel_data = depthwise_mult_batches_weights;
    const q7_t *input_data = depthwise_mult_batches_input;

    input_dims.n = DEPTHWISE_MULT_BATCHES_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_MULT_BATCHES_INPUT_W;
    input_dims.h = DEPTHWISE_MULT_BATCHES_INPUT_H;
    input_dims.c = DEPTHWISE_MULT_BATCHES_IN_CH;
    filter_dims.w = DEPTHWISE_MULT_BATCHES_FILTER_X;
    filter_dims.h = DEPTHWISE_MULT_BATCHES_FILTER_Y;
    output_dims.w = DEPTHWISE_MULT_BATCHES_OUTPUT_W;
    output_dims.h = DEPTHWISE_MULT_BATCHES_OUTPUT_H;
    output_dims.c = DEPTHWISE_MULT_BATCHES_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_MULT_BATCHES_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_MULT_BATCHES_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_MULT_BATCHES_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_MULT_BATCHES_STRIDE_Y;
    dw_conv_params.ch_mult = DEPTHWISE_MULT_BATCHES_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_MULT_BATCHES_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_MULT_BATCHES_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_MULT_BATCHES_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_MULT_BATCHES_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_mult_batches_output_mult;
    quant_params.shift = (int32_t *)depthwise_mult_batches_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

    arm_status result = arm_depthwise_conv_s8(&ctx,
                                              &dw_conv_params,
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
    TEST_ASSERT_TRUE(validate(output, depthwise_mult_batches_output_ref, DEPTHWISE_MULT_BATCHES_DST_SIZE));
}
