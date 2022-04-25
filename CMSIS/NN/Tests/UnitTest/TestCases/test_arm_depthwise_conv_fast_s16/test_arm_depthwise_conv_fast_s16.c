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

#include <arm_nnfunctions.h>
#include <unity.h>

#include "../TestData/dw_int16xint8_fast/test_data.h"
#include "../TestData/dw_int16xint8_fast_spill/test_data.h"
#include "../TestData/dw_int16xint8_fast_stride/test_data.h"
#include "../Utils/validate.h"

void dw_int16xint8_fast_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    q15_t output[DW_INT16XINT8_FAST_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q63_t *bias_data = dw_int16xint8_fast_biases;
    const q15_t *input_data = dw_int16xint8_fast_input;
    const q7_t *kernel_data = dw_int16xint8_fast_weights;
    const q15_t *output_ref = dw_int16xint8_fast_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));

    buf_size = arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void dw_int16xint8_fast_spill_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    q15_t output[DW_INT16XINT8_FAST_SPILL_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q63_t *bias_data = dw_int16xint8_fast_spill_biases;
    const q15_t *input_data = dw_int16xint8_fast_spill_input;
    const q7_t *kernel_data = dw_int16xint8_fast_spill_weights;
    const q15_t *output_ref = dw_int16xint8_fast_spill_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_SPILL_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_SPILL_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_SPILL_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_SPILL_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_SPILL_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_SPILL_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_SPILL_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_SPILL_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_SPILL_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_SPILL_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_SPILL_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_SPILL_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_SPILL_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_SPILL_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_SPILL_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_SPILL_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_SPILL_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_SPILL_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_SPILL_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_SPILL_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_SPILL_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_spill_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_spill_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));

    buf_size = arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}

void dw_int16xint8_fast_stride_arm_depthwise_conv_fast_s16(void)
{
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    q15_t output[DW_INT16XINT8_FAST_STRIDE_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q63_t *bias_data = dw_int16xint8_fast_stride_biases;
    const q15_t *input_data = dw_int16xint8_fast_stride_input;
    const q7_t *kernel_data = dw_int16xint8_fast_stride_weights;
    const q15_t *output_ref = dw_int16xint8_fast_stride_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_FAST_STRIDE_DST_SIZE;

    input_dims.n = DW_INT16XINT8_FAST_STRIDE_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_FAST_STRIDE_INPUT_W;
    input_dims.h = DW_INT16XINT8_FAST_STRIDE_INPUT_H;
    input_dims.c = DW_INT16XINT8_FAST_STRIDE_IN_CH;
    filter_dims.w = DW_INT16XINT8_FAST_STRIDE_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FAST_STRIDE_FILTER_Y;
    output_dims.w = DW_INT16XINT8_FAST_STRIDE_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_FAST_STRIDE_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_FAST_STRIDE_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_FAST_STRIDE_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_FAST_STRIDE_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_FAST_STRIDE_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_FAST_STRIDE_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_FAST_STRIDE_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_FAST_STRIDE_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_FAST_STRIDE_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_FAST_STRIDE_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_FAST_STRIDE_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_FAST_STRIDE_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_FAST_STRIDE_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_fast_stride_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_fast_stride_output_shift;

    int buf_size = arm_depthwise_conv_fast_s16_get_buffer_size(&input_dims, &filter_dims);
    ctx.buf = malloc(buf_size);

    arm_cmsis_nn_status result = arm_depthwise_conv_fast_s16(&ctx,
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
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));

    buf_size = arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    result = arm_depthwise_conv_wrapper_s16(&ctx,
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
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, output_ref_size));
}
