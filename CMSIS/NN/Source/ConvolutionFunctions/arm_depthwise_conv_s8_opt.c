/*
 * Copyright (C) 2010-2019 Arm Limited or its affiliates. All rights reserved.
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

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_depthwise_conv_s8_opt.c
 * Description:  Optimized s8 depthwise separable convolution function for
 *               channel multiplier of 1.
 *
 * $Date:        August 2019
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnsupportfunctions.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

  /*
   * Optimized s8 depthwise convolution function with constraint that in_channel equals out_channel
   *
   *  Refer prototype header file for details.
   *
   */
arm_status arm_depthwise_conv_s8_opt(const q7_t *input,
                                     const uint16_t input_x,
                                     const uint16_t input_y,
                                     const uint16_t input_ch,
                                     const q7_t *kernel,
                                     const uint16_t output_ch,
                                     const uint16_t kernel_x,
                                     const uint16_t kernel_y,
                                     const uint16_t pad_x,
                                     const uint16_t pad_y,
                                     const uint16_t stride_x,
                                     const uint16_t stride_y,
                                     const q7_t *bias,
                                     q7_t *output,
                                     const int32_t *output_shift,
                                     const int32_t *output_mult,
                                     const uint16_t output_x,
                                     const uint16_t output_y,
                                     const int32_t output_offset,
                                     const int32_t input_offset,
                                     const int32_t output_activation_min,
                                     const int32_t output_activation_max,
                                     const uint16_t dilation_x,
                                     const uint16_t dilation_y,
                                     q15_t *buffer_a)
{


    /* Check input constraints input_ch == output_ch */
    if (input_ch != output_ch)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }

#if defined(ARM_MATH_LOOPUNROLL) && defined(ARM_MATH_DSP)
    /* Run the following code in cores using DSP extension */
    (void)dilation_x;
    (void)dilation_y;

    int16_t i_out_y, i_out_x;
    int16_t i_ker_y, i_ker_x;
    q15_t *const col_buffer_start = buffer_a;
    q15_t *col_buffer = col_buffer_start;
    const q7_t *const bias_start_pos = bias;
    const q31_t *const out_mult_start_pos = output_mult;
    const q31_t *const out_shift_start_pos = output_shift;
    uint16_t row_count;
    uint16_t row_shift;


    for (i_out_y = 0; i_out_y < output_y; i_out_y++)
    {
        const int16_t base_idx_y = (i_out_y * stride_y) - pad_y;
        for (i_out_x = 0; i_out_x < output_x; i_out_x++)
        {
            const int16_t base_idx_x = (i_out_x * stride_x) - pad_x;

            /* Out of bounds is only considered for the y axis as it provides a contiguous zero'ing opportunity than along
               the x axis */
            const int ker_y_start = MAX(0, -base_idx_y);
            /* Condition for kernel end dimension: (base_idx_y + ker_y_end) < input_y */
            const int ker_y_end = MIN(input_y, input_y - base_idx_y);

            int32_t index = 0;
            if (ker_y_start != 0)
            {
                memset(&col_buffer[index], 0, (kernel_x * input_ch) * ker_y_start * sizeof(q15_t));
                index += (kernel_x * input_ch) * ker_y_start;
            }

            for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
            {
                const int32_t idx_y = base_idx_y + i_ker_y;

                for (i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                {
                    const int32_t idx_x = base_idx_x + i_ker_x;
                    if (idx_x < 0 || idx_x >= input_x)
                    {
                        memset(&col_buffer[index], 0, input_ch * sizeof(q15_t));
                    }
                    else
                    {
                        arm_q7_to_q15_with_offset((q7_t *)input + (idx_y * input_x + idx_x) * input_ch, &col_buffer[index], input_ch, input_offset);
                    }
                    index += input_ch;
                }
            }

            const int diff = kernel_y - ker_y_end;
            if (diff != 0)
            {
                memset(&col_buffer[index], 0, (kernel_x * input_ch) * diff * sizeof(q15_t));
            }

            row_count = output_ch / 4;
            row_shift = 0;
            bias = bias_start_pos;
            output_mult = out_mult_start_pos;
            output_shift = out_shift_start_pos;

            while (row_count)
            {
                q31_t sum = (q31_t)(*bias++);
                q31_t sum_2 = (q31_t)(*bias++);
                q31_t sum_3 = (q31_t)(*bias++);
                q31_t sum_4 = (q31_t)(*bias++);

                uint16_t col_count = (kernel_x * kernel_y) / 2;
                q15_t *col_pos = col_buffer_start + row_shift;
                const q7_t *row_pos = kernel + row_shift;
                row_shift += 4;

                while (col_count)
                {
                    /* General idea is to read 4 + 4 (input, kernel) pair and re-arrange them in the right order to
                    use in a SMLAD instruction . One run of this loop produces 4 partial outputs with 8 MACs. */
                    /* Note: variable names can be improved here to align with rows and columns. */
                    q31_t ip_a1, ip_a2, ip_b1, ip_b2, op_a, op_b, op_c;
                    /* Read 4 weights */
                    ip_b1 = arm_nn_read_q7x4(row_pos);
                    row_pos += input_ch;
                    ip_a2 = __SXTB16(ip_b1);
                    ip_b1 = __SXTB16(__ROR(ip_b1, 8));
                    ip_a1 = arm_nn_read_q7x4(row_pos);
                    row_pos += input_ch;
                    ip_b2 = __SXTB16(ip_a1);
                    ip_a1 = __SXTB16(__ROR(ip_a1, 8));
                    /* Read channel 0 and channel 1 for col position N and N + 1 */
                    op_a = arm_nn_read_q15x2(col_pos);
                    op_b = arm_nn_read_q15x2(col_pos + input_ch);
                    op_c = __PKHBT(op_a, op_b, 16);
                    op_a = __PKHTB(op_a, op_b, 16);
                    op_b = __PKHBT(ip_a2, ip_b2, 16);
                    sum = __SMLAD(op_c, op_b, sum);
                    op_b = __PKHBT(ip_a1, ip_b1, 16);
                    sum_2 = __SMLAD(op_b, op_a, sum_2);
                    ip_a2 = __PKHTB(ip_a2, ip_b2, 16);
                    /* Read channel 2 and channel 3 for col position N and N + 1 */
                    op_a = arm_nn_read_q15x2(col_pos + 2);
                    col_pos += input_ch;
                    op_b = arm_nn_read_q15x2(col_pos + 2);
                    ip_b2 = __PKHTB(ip_a1, ip_b1, 16);
                    ip_b1 = __PKHBT(op_b, op_a, 16);
                    sum_3 = __SMLAD(ip_b1, ip_a2, sum_3);
                    op_b = __PKHTB(op_b, op_a, 16);
                    sum_4 = __SMLAD(op_b, ip_b2, sum_4);
                    col_pos += input_ch;
                    col_count--;
                }

                col_count = (kernel_x * kernel_y) & 0x1;
                while (col_count)
                {
                    q31_t ip_a2, ip_b1, op_a, op_b;
                    ip_b1 = arm_nn_read_q7x4(row_pos);
                    row_pos += input_ch;

                    ip_a2 = __SXTB16(ip_b1);
                    ip_b1 = __SXTB16(__ROR(ip_b1, 8));

                    op_a = arm_nn_read_q15x2(col_pos);
                    op_b = arm_nn_read_q15x2(col_pos + 2);
                    col_pos += input_ch;

                    sum += (ip_a2 & 0xFFFF) * (op_a & 0xFFFF);
                    op_a >>= 16;
                    sum_2 += (ip_b1 & 0xFFFF) * (op_a);
                    ip_a2 >>= 16;
                    ip_b1 >>= 16;
                    sum_3 += (ip_a2) * (op_b & 0xFFFF);
                    op_b >>= 16;
                    sum_4 += (ip_b1) * (op_b);

                    col_count--;
                }
                sum = arm_nn_requantize(sum, *output_mult++, *output_shift++);
                sum += output_offset;
                sum = MAX(sum, output_activation_min);
                sum = MIN(sum, output_activation_max);
                *output++ = (q7_t)sum;

                sum_2 = arm_nn_requantize(sum_2, *output_mult++, *output_shift++);
                sum_2 += output_offset;
                sum_2 = MAX(sum_2, output_activation_min);
                sum_2 = MIN(sum_2, output_activation_max);
                *output++ = (q7_t)sum_2;
                sum_3 = arm_nn_requantize(sum_3, *output_mult++, *output_shift++);
                sum_3 += output_offset;
                sum_3 = MAX(sum_3, output_activation_min);
                sum_3 = MIN(sum_3, output_activation_max);
                *output++ = (q7_t)sum_3;

                sum_4 = arm_nn_requantize(sum_4, *output_mult++, *output_shift++);
                sum_4 += output_offset;
                sum_4 = MAX(sum_4, output_activation_min);
                sum_4 = MIN(sum_4, output_activation_max);
                *output++ = (q7_t)sum_4;

                row_count--;
            }

            row_count = output_ch & 0x3;
            while (row_count)
            {
                q15_t *col_pos = col_buffer_start + row_shift;
                const q7_t *row_pos = kernel + row_shift;
                q31_t sum = (q31_t)*bias++;
                const uint16_t col_count = (kernel_x * kernel_y);
                row_shift += 1;

                for (int i = 0; i < col_count; i++)
                {
                    sum += row_pos[i * input_ch] * col_pos[i * input_ch];
                }
                sum = arm_nn_requantize(sum, *output_mult++, *output_shift++);
                sum += output_offset;
                sum = MAX(sum, output_activation_min);
                sum = MIN(sum, output_activation_max);
                *output++ = (q7_t)sum;

                row_count--;
            }

            // clear counter and pointers
            col_buffer = col_buffer_start;
        }
    }

#else
    (void)buffer_a;
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    return arm_depthwise_conv_s8(input,
                                 input_x,
                                 input_y,
                                 input_ch,
                                 kernel,
                                 output_ch,
                                 1,
                                 kernel_x,
                                 kernel_y,
                                 pad_x,
                                 pad_y,
                                 stride_x,
                                 stride_y,
                                 bias,
                                 output,
                                 output_shift,
                                 output_mult,
                                 output_x,
                                 output_y,
                                 output_offset,
                                 input_offset,
                                 output_activation_min,
                                 output_activation_max,
                                 dilation_x,
                                 dilation_y,
                                 NULL);
#endif /* ARM_MATH_DSP & ARM_MATH_LOOPUNROLL*/

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of NNConv group
 */
