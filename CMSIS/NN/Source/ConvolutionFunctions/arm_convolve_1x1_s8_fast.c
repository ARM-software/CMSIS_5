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
 * Title:        arm_convolve_1x1_s8_fast.c
 * Description:  Fast q7 version of 1x1 convolution (non-square shape)
 *
 * $Date:        18 December 2019
 * $Revision:    V.1.0.0
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
    if (input_ch % 4 != 0 || output_ch % 2 != 0 ||
        pad_x != 0 || pad_y != 0 ||
        stride_x != 1 || stride_y != 1)
    {
        return ARM_MATH_SIZE_MISMATCH;
    }
#if defined(ARM_MATH_MVEI)
    (void)buffer_a;
    /* Process 4 * N input elements */
    output = arm_nn_mat_mult_s8(kernel,
                                input,
                                output_ch,
                                input_ch,
                                output_shift,
                                output_mult,
                                out_offset,
                                input_offset,
                                0,
                                out_activation_min,
                                out_activation_max,
                                input_x * input_y * input_batches,
                                bias,
                                output);

    const int32_t num_elements = input_x * input_y * input_batches;
    input += (num_elements & ~3) * input_ch;
    for (int i_items = 0; i_items < (num_elements & 3); i_items++)
    {
        for (int i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
        {
            const int8_t *ip_n_0 = input + i_items * input_ch;
            const int8_t *ker_n_0 = kernel + i_out_ch * input_ch;
            int32_t acc = bias[i_out_ch];
            int col_loop = input_ch / 16;
            int32_t off = 0;
            int32_t sum_k = 0;

            while (col_loop > 0)
            {
                const int8x16_t k_0 = vldrbq_s8(ker_n_0 + off);
                sum_k += vaddvq_s8(k_0);
                const int8x16_t n_0 = vldrbq_s8(ip_n_0 + off);
                acc += vmladavq_s8(n_0, k_0);
                off += 16;
                col_loop--;
            }

            col_loop = (input_ch & 0xF);
            if (col_loop != 0)
            {
                const mve_pred16_t p = vctp8q(col_loop);
                const int8x16_t k_0 = vldrbq_z_s8(ker_n_0 + off, p);
                sum_k += vaddvq_p_s8(k_0, p);

                const int8x16_t n_0 = vldrbq_z_s8(ip_n_0 + off, p);
                acc += vmladavq_p_s8(n_0, k_0, p);
            }

            sum_k = (sum_k * input_offset);
            acc += sum_k;
            acc = arm_nn_requantize(acc, output_mult[i_out_ch], output_shift[i_out_ch]);
            acc +=  out_offset;

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
