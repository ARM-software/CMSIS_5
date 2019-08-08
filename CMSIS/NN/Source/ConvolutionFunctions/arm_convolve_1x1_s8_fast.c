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
 * $Date:        August 2019
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/**
   * @brief Fast s8 version for 1x1 convolution (non-square shape)
   *
   * @note  Refer header file for details. Optimal use case for the DSP implementation is when input and output channels
   *        are multiples of 4 or atleast greater than 4.
   *
   */

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

arm_status arm_convolve_1x1_s8_fast(const q7_t *input,
                                    const uint16_t input_x,
                                    const uint16_t input_y,
                                    const uint16_t input_ch,
                                    const q7_t *kernel,
                                    const uint16_t output_ch,
                                    const uint16_t pad_x,
                                    const uint16_t pad_y,
                                    const uint16_t stride_x,
                                    const uint16_t stride_y,
                                    const q7_t *bias,
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
    /* The constraints on padding and stride simplifies the creation of im2col buffer */
   if (pad_x != 0 || pad_y != 0 || stride_x != 1 || stride_y != 1)
    {
       return ARM_MATH_SIZE_MISMATCH;
    }

#if defined(ARM_MATH_LOOPUNROLL) && defined(ARM_MATH_DSP)
    /* Optimized version for M cores with DSP extension */
    int16_t i_out_y, i_out_x;
    int16_t i_ch_out;

    /* Partial(two columns) im2col buffer */
    q15_t *two_column_buffer = buffer_a;
    q7_t *out = output;

    for (i_out_y = 0; i_out_y < output_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < output_x; i_out_x++)
        {
            /* Fill buffer for partial im2col */
            arm_q7_to_q15_reordered_with_offset(input + (i_out_y * input_x + i_out_x) * input_ch,
                                                two_column_buffer,
                                                input_ch,
                                                (q7_t)input_offset);
            two_column_buffer += input_ch;

            if (two_column_buffer == buffer_a + 2 * input_ch * DIM_KER_X * DIM_KER_Y)
            {
                out = arm_nn_mat_mult_kernel_s8_s16_reordered(kernel,
                                                              buffer_a,
                                                              output_ch,
                                                              output_shift,
                                                              output_mult,
                                                              (q7_t)out_offset,
                                                              out_activation_min,
                                                              out_activation_max,
                                                              input_ch * DIM_KER_Y * DIM_KER_X,
                                                              bias, out);
                /* counter reset */
                two_column_buffer = buffer_a;
            }
        }
    }

    /* check if there is an odd column left-over for computation */
    if (two_column_buffer != buffer_a)
    {
        const q7_t *ker_a = kernel;
        for (i_ch_out = 0; i_ch_out < output_ch; i_ch_out++)
        {
            q31_t sum = (q31_t)bias[i_ch_out];

            /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
            q15_t *ip_as_col = buffer_a;
            uint16_t col_count = (input_ch * DIM_KER_X * DIM_KER_Y) >> 2;

            while (col_count)
            {
                q31_t ker_a1, ker_a2;
                q31_t in_b1, in_b2;
                ker_a = (const q7_t *)read_and_pad_reordered((void *)ker_a, &ker_a1, &ker_a2);

                in_b1 = arm_nn_read_q15x2_ia((const q15_t **)&ip_as_col);
                sum = __SMLAD(ker_a1, in_b1, sum);
                in_b2 = arm_nn_read_q15x2_ia((const q15_t **)&ip_as_col);
                sum = __SMLAD(ker_a2, in_b2, sum);

                col_count--;
            }
            col_count = input_ch * DIM_KER_Y * DIM_KER_X & 0x3;
            while (col_count)
            {
                q7_t ker_a1 = *ker_a++;
                q15_t in_b1 = *ip_as_col++;
                sum += ker_a1 * in_b1;
                col_count--;
            }
            sum = arm_nn_requantize(sum, output_mult[i_ch_out], output_shift[i_ch_out]);
            sum += out_offset;
            sum = MAX(sum, out_activation_min);
            sum = MIN(sum, out_activation_max);
            *out++ = (q7_t)sum;
        }
    }

#else
    /* Run the following code as reference implementation for M cores with no DSP extension or when loop unrolling is
       not to be done */
    (void)buffer_a;
    return arm_convolve_s8(input, input_x, input_y,
                           input_ch, kernel, output_ch,
                           DIM_KER_X, DIM_KER_Y,
                           pad_x, pad_y,
                           stride_x, stride_y,
                           bias, output,
                           output_shift, output_mult,
                           out_offset, input_offset,
                           out_activation_min, out_activation_max,
                           output_x, output_y,
                           NULL);
#endif

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of NNConv group
 */
