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

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_1x1_s8_fast.c
 * Description:  Fast q7 version of 1x1 convolution (non-square shape)
 *
 * $Date:        January 20, 2020
 * $Revision:    V.1.0.2
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
   * Fast s8 version for 1x1 convolution (non-square shape)
   *
   * Refer header file for details.
   *
   */

arm_status arm_convolve_1x1_s8_fast(const q7_t *input,
                                    const uint16_t input_x,
                                    const uint16_t input_y,
                                    const uint16_t input_ch,
                                    const uint16_t input_batches,
                                    const q7_t *kernel,
                                    const uint16_t output_ch,
                                    const uint16_t pad_x,
                                    const uint16_t pad_y,
                                    const uint16_t stride_x,
                                    const uint16_t stride_y,
                                    const int32_t *bias,
                                    q7_t *output,
                                    const int32_t *output_shift,
                                    const int32_t *output_mult,
                                    const int32_t out_offset,
                                    const int32_t input_offset,
                                    const int32_t out_activation_min,
                                    const int32_t out_activation_max,
                                    const uint16_t output_x,
                                    const uint16_t output_y,
                                    q15_t *buffer_a)
{
    if (input_ch % 4 != 0 ||
        pad_x != 0 || pad_y != 0 ||
        stride_x != 1 || stride_y != 1)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }
#if defined(ARM_MATH_MVEI)
    (void)buffer_a;

    int32_t col_len = input_x * input_y * input_batches;

    for (int i_items = 0; i_items <= (col_len - 4); i_items += 4)
    {
        for (int i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
        {
            int32_t sum_row = 0;
            int32_t temp_out[4];

            (void)arm_nn_mat_mul_core_4x_s8(input_ch,
                                            input + i_items * input_ch,
                                            kernel + i_out_ch * input_ch,
                                            &sum_row,
                                            temp_out);
            int32x4_t res = vldrwq_s32(temp_out);

            res = vaddq_n_s32(res, bias[i_out_ch]);
            sum_row = sum_row * input_offset;
            res = vaddq_n_s32(res, sum_row);
            res = arm_requantize_mve(res, output_mult[i_out_ch], output_shift[i_out_ch]);
            res = vaddq_n_s32(res, out_offset);

            res = vmaxq_s32(res, vdupq_n_s32(out_activation_min));
            res = vminq_s32(res, vdupq_n_s32(out_activation_max));

            const uint32x4_t scatter_offset = {0, output_ch, output_ch * 2, output_ch * 3};
            vstrbq_scatter_offset_s32(output, scatter_offset, res);
            output++;
        }
        output += (3 * output_ch);
    }

    /* Handle left over elements */
    for (int i_items = (col_len & ~0x3); i_items < col_len; i_items++)
    {
        for (int i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
        {
            int32_t sum_row = 0;

            int32_t acc;
            (void)arm_nn_mat_mul_core_1x_s8(input_ch,
                                            input + i_items * input_ch,
                                            kernel + i_out_ch * input_ch,
                                            &sum_row,
                                            &acc);

            acc += bias[i_out_ch];
            sum_row = (sum_row * input_offset);
            acc += sum_row;
            acc = arm_nn_requantize(acc, output_mult[i_out_ch], output_shift[i_out_ch]);
            acc += out_offset;

            acc = MAX(acc, out_activation_min);
            acc = MIN(acc, out_activation_max);
            *output++ = acc;
        }
    }


#elif defined(ARM_MATH_LOOPUNROLL) && defined(ARM_MATH_DSP)
    int32_t i_element;
    (void)input_x;
    (void)input_y;

    /* Partial(two columns) im2col buffer */
    q15_t *two_column_buffer = buffer_a;
    q7_t *out = output;
    const int32_t num_elements = output_x * output_y * input_batches;

    for (i_element = 0; i_element < num_elements / 2; i_element++)
    {
        /* Fill buffer for partial im2col - two columns at a time */
        arm_q7_to_q15_reordered_with_offset(&input[i_element * 2 * input_ch],
                                            two_column_buffer,
                                            input_ch * 2,
                                            input_offset);

        out = arm_nn_mat_mult_kernel_s8_s16_reordered(kernel,
                                                      two_column_buffer,
                                                      output_ch,
                                                      output_shift,
                                                      output_mult,
                                                      (q7_t)out_offset,
                                                      out_activation_min,
                                                      out_activation_max,
                                                      input_ch * DIM_KER_Y * DIM_KER_X,
                                                      bias, out);
    }

    /* check if there is an odd column left-over for computation */
    if (num_elements & 0x1)
    {
        int32_t i_ch_out;
        const q7_t *ker_a = kernel;

        arm_q7_to_q15_reordered_with_offset(
            &input[(num_elements - 1) * input_ch],
            two_column_buffer, input_ch, input_offset);

        for (i_ch_out = 0; i_ch_out < output_ch; i_ch_out++)
        {
            q31_t sum = bias[i_ch_out];

            /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
            const q15_t *ip_as_col = buffer_a;
            uint16_t col_count = (input_ch * DIM_KER_X * DIM_KER_Y) >> 2;

            while (col_count)
            {
                q31_t ker_a1, ker_a2;
                q31_t in_b1, in_b2;
                ker_a = read_and_pad_reordered(ker_a, &ker_a1, &ker_a2);

                in_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
                sum = __SMLAD(ker_a1, in_b1, sum);
                in_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
                sum = __SMLAD(ker_a2, in_b2, sum);

                col_count--;
            }

            sum = arm_nn_requantize(sum, output_mult[i_ch_out],
                                    output_shift[i_ch_out]);
            sum += out_offset;
            sum = MAX(sum, out_activation_min);
            sum = MIN(sum, out_activation_max);
            *out++ = (q7_t)sum;
        }
    }

#else
    /* Run the following code as reference implementation for M cores with no DSP extension or when loop unrolling is
       not to be done */
    return arm_convolve_s8(input, input_x, input_y,
                           input_ch, input_batches, kernel, output_ch,
                           DIM_KER_X, DIM_KER_Y,
                           pad_x, pad_y,
                           stride_x, stride_y,
                           bias, output,
                           output_shift, output_mult,
                           out_offset, input_offset,
                           out_activation_min, out_activation_max,
                           output_x, output_y,
                           buffer_a);
#endif

    /* Return to application */
    return ARM_MATH_SUCCESS;
}


int32_t arm_convolve_1x1_s8_fast_get_buffer_size(const uint16_t input_ch)
{
#if defined(ARM_MATH_LOOPUNROLL) && defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    return 2 * input_ch * sizeof(int16_t);
#else
    (void)input_ch;
    return 0;
#endif
}

/**
 * @} end of NNConv group
 */
