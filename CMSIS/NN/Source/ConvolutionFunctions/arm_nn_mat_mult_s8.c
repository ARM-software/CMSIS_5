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
 * Title:        arm_nn_mat_mult_s8.c
 * Description:  General Matrix-multiplication function
 *
 * $Date:        November 2019
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/*
   * s8 General matrix multiplication function with per-channel requantization.
   *
   * Refer header file for details.
   *
   */

q7_t *arm_nn_mat_mult_s8(const q7_t *input_row,
                         const q7_t *input_col,
                         const uint16_t output_ch,
                         const uint16_t input_ch,
                         const int32_t *output_shift,
                         const int32_t *output_mult,
                         const int32_t out_offset,
                         const int32_t col_offset,
                         const int32_t row_offset,
                         const int16_t activation_min,
                         const int16_t activation_max,
                         const uint16_t col_len,
                         const int32_t *const bias,
                         q7_t *out)
{
#if defined(ARM_MATH_MVEI)

    (void)row_offset;
    for (int i_items = 0; i_items <= (col_len - 4); i_items += 4)
    {
        for (int i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
        {
            q31_t acc_n0 = bias[i_out_ch];
            q31_t acc_n1 = acc_n0;
            q31_t acc_n2 = acc_n0;
            q31_t acc_n3 = acc_n0;
            int col_loop = input_ch / 16;

            const int8_t *ip_n_0 = input_col + i_items * input_ch;
            const int8_t *ip_n_1 = ip_n_0 + input_ch;
            const int8_t *ip_n_2 = ip_n_1 + input_ch;
            const int8_t *ip_n_3 = ip_n_2 + input_ch;

            const int8_t *ker_n_0 = input_row + i_out_ch * input_ch;
            int32_t offset = 0;
            int32_t sum_row = 0;

            while (col_loop > 0)
            {
                const int8x16_t k_0 = vldrbq_s8(ker_n_0 + offset);
                sum_row += vaddvq_s8(k_0);

                const int8x16_t n_0 = vldrbq_s8(ip_n_0 + offset);
                const int8x16_t n_1 = vldrbq_s8(ip_n_1 + offset);
                const int8x16_t n_2 = vldrbq_s8(ip_n_2 + offset);
                const int8x16_t n_3 = vldrbq_s8(ip_n_3 + offset);

                acc_n0 += vmladavq_s8(n_0, k_0);
                acc_n1 += vmladavq_s8(n_1, k_0);
                acc_n2 += vmladavq_s8(n_2, k_0);
                acc_n3 += vmladavq_s8(n_3, k_0);

                offset += 16;
                col_loop--;
            }

            col_loop = (input_ch & 0xF);

            if (col_loop != 0)
            {
                const mve_pred16_t p = vctp8q(col_loop);

                const int8x16_t k_0 = vldrbq_z_s8(ker_n_0 + offset, p);
                sum_row += vaddvq_p_s8(k_0, p);

                const int8x16_t n_0 = vldrbq_z_s8(ip_n_0 + offset, p);
                const int8x16_t n_1 = vldrbq_z_s8(ip_n_1 + offset, p);
                const int8x16_t n_2 = vldrbq_z_s8(ip_n_2 + offset, p);
                const int8x16_t n_3 = vldrbq_z_s8(ip_n_3 + offset, p);

                acc_n0 += vmladavq_p_s8(n_0, k_0, p);
                acc_n1 += vmladavq_p_s8(n_1, k_0, p);
                acc_n2 += vmladavq_p_s8(n_2, k_0, p);
                acc_n3 += vmladavq_p_s8(n_3, k_0, p);
            }
            int32x4_t res;
            res[0] = acc_n0;
            res[1] = acc_n1;
            res[2] = acc_n2;
            res[3] = acc_n3;

            sum_row = sum_row * col_offset;
            res = vaddq_n_s32(res, sum_row);
            res = arm_requantize_mve(res, output_mult[i_out_ch], output_shift[i_out_ch]);
            res = vaddq_n_s32(res, out_offset);

            res = vmaxq_s32(res, vdupq_n_s32(activation_min));
            res = vminq_s32(res, vdupq_n_s32(activation_max));

            out[i_out_ch] = res[0];
            out[i_out_ch + output_ch] = res[1];
            out[i_out_ch + output_ch * 2] = res[2];
            out[i_out_ch + output_ch * 3] = res[3];
        }

        out += (4 * output_ch);
    }

    return out;

#else
    /* TODO: Add support for DSP extension */
    (void)input_row;
    (void)input_col;
    (void)output_ch;
    (void)input_ch;
    (void)output_shift;
    (void)output_mult;
    (void)out_offset;
    (void)col_offset;
    (void)row_offset;
    (void)activation_min;
    (void)activation_max;
    (void)col_len;
    (void)bias;
    (void)out;
    return NULL;
#endif
}