/*
 * SPDX-FileCopyrightText: Copyright 2010-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_convolve_1x1_HWC_q7_fast_nonsquare.c
 * Description:  Fast Q7 version of 1x1 convolution (non-square shape)
 *
 * $Date:        4 Aug 2022
 * $Revision:    V.2.0.1
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

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
 * Fast Q7 version of 1x1 convolution (non-sqaure shape)
 * Refer function header for details
 *
 */

arm_cmsis_nn_status arm_convolve_1x1_HWC_q7_fast_nonsquare(const q7_t *Im_in,
                                                           const uint16_t dim_im_in_x,
                                                           const uint16_t dim_im_in_y,
                                                           const uint16_t ch_im_in,
                                                           const q7_t *wt,
                                                           const uint16_t ch_im_out,
                                                           const uint16_t dim_kernel_x,
                                                           const uint16_t dim_kernel_y,
                                                           const uint16_t padding_x,
                                                           const uint16_t padding_y,
                                                           const uint16_t stride_x,
                                                           const uint16_t stride_y,
                                                           const q7_t *bias,
                                                           const uint16_t bias_shift,
                                                           const uint16_t out_shift,
                                                           q7_t *Im_out,
                                                           const uint16_t dim_im_out_x,
                                                           const uint16_t dim_im_out_y,
                                                           q15_t *bufferA,
                                                           q7_t *bufferB)
{
    (void)bufferB;
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    /* Run the following code for Cortex-M4 and Cortex-M7 */
    (void)dim_im_in_y;
    int16_t i_out_y, i_out_x;
    int16_t i_ch_out;

    /* -----------------------
     *  Here we use bufferA as q15_t internally as computation are done with q15_t level
     *  im2col are done to output in q15_t format from q7_t input
     */

    q15_t *pBuffer = bufferA;
    q7_t *pOut = Im_out;

    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0 || dim_kernel_x != 1 || dim_kernel_y != 1 || padding_x != 0 ||
        padding_y != 0 || stride_x != 1 || stride_y != 1)
    {
        /* check if the input dimension meets the constraints */
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (i_out_y = 0; i_out_y < dim_im_out_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
        {
            /* This part implements the im2col function */
            arm_q7_to_q15_reordered_no_shift(
                (q7_t *)Im_in + (i_out_y * dim_im_in_x + i_out_x) * ch_im_in, pBuffer, ch_im_in);
            pBuffer += ch_im_in;

            if (pBuffer == bufferA + 2 * ch_im_in * dim_kernel_x * dim_kernel_y)
            {
                pOut = arm_nn_mat_mult_kernel_q7_q15_reordered(
                    wt, bufferA, ch_im_out, ch_im_in, bias_shift, out_shift, bias, pOut);
                /* counter reset */
                pBuffer = bufferA;
            }
        }
    }

    /* check if there is left-over for compute */
    if (pBuffer != bufferA)
    {
        const q7_t *pA = wt;
        for (i_ch_out = 0; i_ch_out < ch_im_out; i_ch_out++)
        {
            q31_t sum = ((q31_t)(bias[i_ch_out]) << bias_shift) + NN_ROUND(out_shift);
            const q15_t *pB = bufferA;
            /* basically each time it process 4 entries */
            uint16_t colCnt = ch_im_in * dim_kernel_x * dim_kernel_y >> 2;

            while (colCnt)
            {

                q31_t inA1, inA2;
                q31_t inB1, inB2;

                pA = read_and_pad_reordered(pA, &inA1, &inA2);

                inB1 = arm_nn_read_q15x2_ia(&pB);
                sum = __SMLAD(inA1, inB1, sum);
                inB2 = arm_nn_read_q15x2_ia(&pB);

                sum = __SMLAD(inA2, inB2, sum);

                colCnt--;
            }
            colCnt = ch_im_in * dim_kernel_y * dim_kernel_x & 0x3;
            while (colCnt)
            {
                q7_t inA1 = *pA++;
                q15_t inB1 = *pB++;
                sum += inA1 * inB1;
                colCnt--;
            }
            *pOut = (q7_t)__SSAT((sum >> out_shift), 8);
            pOut++;
        }
    }

#else
    (void)bufferA;
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    int i, j, k, l, m, n;
    int conv_out;
    int in_row, in_col;

    if (ch_im_in % 4 != 0 || ch_im_out % 2 != 0 || dim_kernel_x != 1 || dim_kernel_y != 1 || padding_x != 0 ||
        padding_y != 0 || stride_x != 1 || stride_y != 1)
    {
        /* check if the input dimension meets the constraints */
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out_y; j++)
        {
            for (k = 0; k < dim_im_out_x; k++)
            {
                conv_out = ((q31_t)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
                for (m = 0; m < dim_kernel_y; m++)
                {
                    for (n = 0; n < dim_kernel_x; n++)
                    {
                        // if-for implementation
                        in_row = stride_y * j + m - padding_y;
                        in_col = stride_x * k + n - padding_x;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in_y && in_col < dim_im_in_x)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out += Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in + l] *
                                    wt[i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_y + n) * ch_im_in +
                                       l];
                            }
                        }
                    }
                }
                Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q7_t)__SSAT((conv_out >> out_shift), 8);
            }
        }
    }

#endif /* ARM_MATH_DSP */

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of NNConv group
 */
