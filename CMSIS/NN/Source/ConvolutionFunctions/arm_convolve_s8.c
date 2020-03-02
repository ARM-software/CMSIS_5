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
 * Title:        arm_convolve_s8.c
 * Description:  s8 version of convolution using symmetric quantization.
 *
 * $Date:        March 1, 2020
 * $Revision:    V.1.0.0
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

/*
   * Basic s8 convolution function.
   *
   * Refer header file for details. Optimal use case for the DSP/MVE implementation is when input and output channels
   * are multiples of 4 or atleast greater than 4.
   *
   */

arm_status arm_convolve_s8(const q7_t *input,
                           const uint16_t input_x,
                           const uint16_t input_y,
                           const uint16_t input_ch,
                           const uint16_t input_batches,
                           const q7_t *kernel,
                           const uint16_t output_ch,
                           const uint16_t kernel_x,
                           const uint16_t kernel_y,
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
    int i_batch;
    for (i_batch = 0; i_batch < input_batches; i_batch++)
    {
        input += i_batch * (input_x * input_y * input_ch);
        output += i_batch * (output_x * output_y * output_ch);
#if defined(ARM_MATH_MVEI)
        /* Generate upto four columns from the input tensor a GEMM computation */
        q7_t *im2col_buf = (q7_t *)buffer_a;
        q7_t *out = output;
        int32_t buffer_fill_cnt = 0;
        int32_t padded = 0;
        const int32_t num_elem = kernel_x * kernel_y * input_ch;

        /* This part implements the im2col function */
        for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                for (int i_ker_y = i_out_y * stride_y - pad_y; i_ker_y < i_out_y * stride_y - pad_y + kernel_y; i_ker_y++)
                {
                    for (int i_ker_x = i_out_x * stride_x - pad_x; i_ker_x < i_out_x * stride_x - pad_x + kernel_x; i_ker_x++)
                    {
                        if (i_ker_y < 0 || i_ker_y >= input_y || i_ker_x < 0 || i_ker_x >= input_x)
                        {
                            memset(im2col_buf, (int8_t)-input_offset, sizeof(q7_t) * input_ch);
                            padded = 1;
                        }
                        else
                        {
                            arm_memcpy_q7(im2col_buf, input + (i_ker_y * input_x + i_ker_x) * input_ch, input_ch);
                        }
                        im2col_buf += input_ch;
                    }
                }

                buffer_fill_cnt++;

                /* Computation is filed for every 4 columns */
                if (buffer_fill_cnt == 4 && (padded == 0))
                {
                    buffer_fill_cnt = 0;
                    for (int i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
                    {
                        int32_t sum_row;
                        int32_t acc[4];

                        (void)arm_nn_mat_mul_core_4x_s8(num_elem,
                                                        num_elem,
                                                        (q7_t *)buffer_a,
                                                        kernel + num_elem * i_out_ch,
                                                        &sum_row,
                                                        acc);
                        int32x4_t s_offset = vdupq_n_s32(sum_row);

                        int32x4_t res = vldrwq_s32(acc);
                        s_offset = vmulq_n_s32(s_offset, input_offset);

                        res = vaddq_n_s32(res, bias[i_out_ch]);
                        res = vaddq_s32(res, s_offset);
                        res = arm_requantize_mve(res, output_mult[i_out_ch], output_shift[i_out_ch]);
                        res = vaddq_n_s32(res, out_offset);

                        res = vmaxq_s32(res, vdupq_n_s32(out_activation_min));
                        res = vminq_s32(res, vdupq_n_s32(out_activation_max));

                        const uint32x4_t scatter_offset = {0, output_ch, output_ch * 2, output_ch * 3};
                        vstrbq_scatter_offset_s32(out, scatter_offset, res);
                        out++;
                    }
                    out += (3 * output_ch);
                    im2col_buf = (q7_t *)buffer_a;
                }
                else if (buffer_fill_cnt == 4 && (padded != 0))
                {
                    buffer_fill_cnt = 0;
                    out = arm_nn_mat_mult_s8(kernel,
                                             (q7_t *)buffer_a,
                                             output_ch,
                                             4,
                                             output_shift,
                                             output_mult,
                                             out_offset,
                                             input_offset,
                                             0,
                                             out_activation_min,
                                             out_activation_max,
                                             num_elem,
                                             bias,
                                             out);

                    im2col_buf = (q7_t *)buffer_a;
                    padded = 0;
                }
            }
        }
        /* Handle left over columns */
        if (buffer_fill_cnt != 0)
        {
            out = arm_nn_mat_mult_s8(kernel,
                                     (q7_t *)buffer_a,
                                     output_ch,
                                     buffer_fill_cnt,
                                     output_shift,
                                     output_mult,
                                     out_offset,
                                     input_offset,
                                     0,
                                     out_activation_min,
                                     out_activation_max,
                                     num_elem,
                                     bias,
                                     out);
        }

#elif defined(ARM_MATH_DSP)
        int32_t i_out_y, i_out_x, i_ker_y, i_ker_x;

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
                q31_t sum = bias[i];

                /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
                const q15_t *ip_as_col = buffer_a;

                /* 4 multiply and accumulates are done in one loop. */
                uint16_t col_count = (input_ch * kernel_y * kernel_x) >> 2;

                while (col_count)
                {
                    q31_t ker_a1, ker_a2;
                    q31_t ip_b1, ip_b2;

                    ker_a = read_and_pad(ker_a, &ker_a1, &ker_a2);

                    ip_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
                    sum = __SMLAD(ker_a1, ip_b1, sum);
                    ip_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
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
        (void)buffer_a;
        int32_t i_out_ch, i_out_y, i_out_x, i_input_ch, i_ker_y, i_ker_x;
        int32_t conv_out;

        for (i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
        {
            for (i_out_y = 0; i_out_y < output_y; i_out_y++)
            {
                for (i_out_x = 0; i_out_x < output_x; i_out_x++)
                {
                    conv_out = bias[i_out_ch];

                    const int32_t base_idx_y = stride_y * i_out_y - pad_y;
                    const int32_t base_idx_x = stride_x * i_out_x - pad_x;

                    const int32_t ker_y_start = MAX(0, -base_idx_y);
                    const int32_t ker_x_start = MAX(0, -base_idx_x);

                    const int32_t ker_y_end = MIN(kernel_y, input_y - base_idx_y);
                    const int32_t ker_x_end = MIN(kernel_x, input_x - base_idx_x);

                    for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                    {
                        for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                        {
                            const int32_t in_row = base_idx_y + i_ker_y;
                            const int32_t in_col = base_idx_x + i_ker_x;
                            for (i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
                            {
                                conv_out +=
                                    (input[(in_row * input_x + in_col) * input_ch + i_input_ch] + input_offset) *
                                    kernel[i_out_ch * input_ch * kernel_y * kernel_x +
                                           (i_ker_y * kernel_x + i_ker_x) * input_ch + i_input_ch];
                            }
                        }
                    }
                    conv_out = arm_nn_requantize(conv_out, output_mult[i_out_ch], output_shift[i_out_ch]);
                    conv_out += out_offset;
                    conv_out = MAX(conv_out, out_activation_min);
                    conv_out = MIN(conv_out, out_activation_max);
                    output[i_out_ch + (i_out_y * output_x + i_out_x) * output_ch] = (int8_t)conv_out;
                }
            }
        }
#endif
    }

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

int32_t arm_convolve_s8_get_buffer_size(const uint16_t input_ch,
                                        const uint16_t kernel_x,
                                        const uint16_t kernel_y)
{
#if defined(ARM_MATH_DSP)
    return (2 * input_ch * kernel_x * kernel_y) * sizeof(int16_t);
#else
    (void)input_ch;
    (void)kernel_x;
    (void)kernel_y;
    return 0;
#endif
}

/**
 * @} end of NNConv group
 */
