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

#include "../TestData/kernel1x1/test_data.h"
#include "../Utils/validate.h"

void kernel1x1_arm_convolve_1x1_s8_fast(void)
{
    const arm_status expected = ARM_MATH_SUCCESS;
    q7_t output[KERNEL1X1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const q31_t *bias_data = kernel1x1_biases;
    const q7_t *input_data = kernel1x1_input;

    input_dims.n = KERNEL1X1_INPUT_BATCHES;
    input_dims.w = KERNEL1X1_INPUT_W;
    input_dims.h = KERNEL1X1_INPUT_H;
    input_dims.c = KERNEL1X1_IN_CH;
    filter_dims.w = KERNEL1X1_FILTER_X;
    filter_dims.h = KERNEL1X1_FILTER_Y;
    output_dims.w = KERNEL1X1_OUTPUT_W;
    output_dims.h = KERNEL1X1_OUTPUT_H;
    output_dims.c = KERNEL1X1_OUT_CH;

    conv_params.padding.w = KERNEL1X1_PAD_X;
    conv_params.padding.h = KERNEL1X1_PAD_Y;
    conv_params.stride.w = KERNEL1X1_STRIDE_X;
    conv_params.stride.h = KERNEL1X1_STRIDE_Y;

    conv_params.input_offset = KERNEL1X1_INPUT_OFFSET;
    conv_params.output_offset = KERNEL1X1_OUTPUT_OFFSET;
    conv_params.activation.min = KERNEL1X1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = KERNEL1X1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)kernel1x1_output_mult;
    quant_params.shift = (int32_t *)kernel1x1_output_shift;

    const int32_t buf_size = arm_convolve_1x1_s8_fast_get_buffer_size(&input_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    arm_status result = arm_convolve_1x1_s8_fast(&ctx,
                                                 &conv_params,
                                                 &quant_params,
                                                 &input_dims,
                                                 input_data,
                                                 &filter_dims,
                                                 kernel1x1_weights,
                                                 &bias_dims,
                                                 bias_data,
                                                 &output_dims,
                                                 output);

    free(ctx.buf);
    TEST_ASSERT_EQUAL(expected, result);
    TEST_ASSERT_TRUE(validate(output, kernel1x1_output_ref, KERNEL1X1_DST_SIZE));
}
