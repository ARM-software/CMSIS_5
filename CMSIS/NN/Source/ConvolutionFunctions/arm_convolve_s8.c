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
 * Title:        arm_convolve_s8.c
 * Description:	 s8 version of convolution using symmetric quantization.
 *
 * $Date:        July 2019
 * $Revision:    V.0.0.1
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/**
   * @brief Basic s8 convolution function.
   *
   * @note  Refer header file for details.
   *
   */

arm_status arm_convolve_s8(const q7_t *input,
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
                           const int32_t out_offset,
                           const int32_t input_offset,
                           const int32_t out_activation_min,
                           const int32_t out_activation_max,
                           const uint16_t output_x,
                           const uint16_t output_y,
                           q15_t *buffer_a)
{

#if defined(ARM_MATH_LOOPUNROLL) && defined (ARM_MATH_DSP)
    int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;

    /* Generate two columns from the input tensor a GEMM computation */
    q15_t *two_column_buf = buffer_a;
    q7_t *out = output;

    /* This part implements the im2col function */
    for (i_out_y = 0; i_out_y < output_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < output_x; i_out_x++)
        {
            for (i_ker_y = i_out_y * stride_y - pad_y; i_ker_y < i_out_y * stride_y - pad_y + kernel_y; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride_x - pad_x; i_ker_x < i_out_x * stride_x - pad_x + kernel_x; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= input_y || i_ker_x < 0 || i_ker_x >= input_x)
                    {
                        /* Filling 0 for out-of-bound paddings */
                        memset(two_column_buf, 0, sizeof(q15_t) * input_ch);
                    }
                    else
                    {
                        /* Copying the pixel data to column */
                        arm_q7_to_q15_with_offset(input + (i_ker_y * input_x + i_ker_x) * input_ch, two_column_buf, input_ch, input_offset);
                    }
                    two_column_buf += input_ch;
                }
            }

            /* Computation is filed for every 2 columns */
            if (two_column_buf == buffer_a + 2 * input_ch * kernel_y * kernel_x)
            {
                out =
                    arm_nn_mat_mult_kernel_s8_s16(kernel,
                                                  buffer_a,
                                                  output_ch,
                                                  output_shift,
                                                  output_mult,
                                                  out_offset,
                                                  out_activation_min,
                                                  out_activation_max,
                                                  input_ch * kernel_y * kernel_x,
                                                  bias,
                                                  out);

                /* counter reset */
                two_column_buf = buffer_a;
            }
        }
    }

    /* left-over because odd number of output pixels */
    if (two_column_buf != buffer_a)
    {
        const q7_t *ker_a = kernel;
        int i;

        for (i = 0; i < output_ch; i++)
        {
            /* Load the accumulator with bias first */
            q31_t sum = (q31_t)bias[i];

            /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
            q15_t *ip_as_col = buffer_a;

            /* 4 multiply and accumulates are done in one loop. */
            uint16_t col_count = (input_ch * kernel_y * kernel_x) >> 2;

            while (col_count)
            {
                q31_t ker_a1, ker_a2;
                q31_t ip_b1, ip_b2;

                ker_a = (q7_t *)read_and_pad((void *)ker_a, &ker_a1, &ker_a2);

                ip_b1 = arm_nn_read_q15x2_ia((const q15_t **)&ip_as_col);
                sum = __SMLAD(ker_a1, ip_b1, sum);
                ip_b2 = arm_nn_read_q15x2_ia((const q15_t **)&ip_as_col);
                sum = __SMLAD(ker_a2, ip_b2, sum);

                col_count--;
            }
            /* Handle left over mac */
            col_count = input_ch * kernel_y * kernel_x & 0x3;
            while (col_count)
            {
                q7_t ker_a1 = *ker_a++;
                q15_t ip_b1 = *ip_as_col++;
                sum += ker_a1 * ip_b1;
                col_count--;
            }

            sum = arm_nn_requantize(sum, output_mult[i], output_shift[i]);
            sum += out_offset;
            sum = MAX(sum, out_activation_min);
            sum = MIN(sum, out_activation_max);
            *out++ = (q7_t)sum;
        }
    }
#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    uint16_t i, j, k, l, m, n;
    int conv_out;
    signed char in_row, in_col;

    for (i = 0; i < output_ch; i++)
    {
        for (j = 0; j < output_y; j++)
        {
            for (k = 0; k < output_x; k++)
            {
                conv_out = bias[i];
                for (m = 0; m < kernel_y; m++)
                {
                    for (n = 0; n < kernel_x; n++)
                    {
                        // if-for implementation
                        in_row = stride_y * j + m - pad_y;
                        in_col = stride_x * k + n - pad_x;
                        if (in_row >= 0 && in_col >= 0 && in_row < input_y && in_col < input_x)
                        {
                            for (l = 0; l < input_ch; l++)
                            {
                                conv_out +=
                                    (input[(in_row * input_x + in_col) * input_ch + l] + input_offset) *
                                    kernel[i * input_ch * kernel_y * kernel_x +
                                           (m * kernel_x + n) * input_ch + l];
                            }
                        }
                    }
                }
                conv_out = arm_nn_requantize(conv_out, output_mult[i], output_shift[i]);
                conv_out += out_offset;
                conv_out = MAX(conv_out, out_activation_min);
                conv_out = MIN(conv_out, out_activation_max);
                output[i + (j * output_x + k) * output_ch] = (q7_t)conv_out;
            }
        }
    }

#endif

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of NNConv group
 */
