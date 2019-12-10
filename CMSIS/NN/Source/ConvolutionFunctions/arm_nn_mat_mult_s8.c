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
#include "arm_nnsupportfunctions.h"
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
                sum_row += vaddvq_p_s32(k_0, p);

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

        out += ((i_items + 1) * 4 * output_ch);
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

/*
 * s8 General matrix multiplication function with per-channel requantization.
 *
 * This function assumes:
 * LHS input matrix NOT transposed
 * RHS input matrix transposed
 *
 * Refer header file for details.
 *
 */

q7_t * arm_nn_mat_mult_nt_t_s8(const q7_t *lhs,
                               const q7_t *rhs,
                               const int32_t *bias,
                               const int32_t *dst_multipliers,
                               const int32_t *dst_shifts,
                               const int32_t m,
                               const int32_t n,
                               const int32_t k,
                               const int32_t lhs_offset,
                               const int32_t dst_offset,
                               const int32_t activation_min,
                               const int32_t activation_max,
                               q7_t *dst) {

#if defined(ARM_MATH_LOOPUNROLL) && defined(ARM_MATH_DSP)
    // Check input constraint. "n" must be a multiple of two
    if (n % 2)
    {
        return NULL;
    }

    // Run the following code for Cortex-M4 and Cortex-M7
    const int32_t off0 = k - 4;

    for (int32_t n_idx = 0; n_idx < n; n_idx+=2)
    {
        const q7_t *lhs_ptr = &lhs[0];
        q7_t *dst_ptr = &dst[0];

        int32_t offset_contribution0 = 0;
        int32_t offset_contribution1 = 0;

        // Calculate the offset contribution
        for(int32_t x = 0; x < k; ++x)
        {
            offset_contribution0 += rhs[x];
            offset_contribution1 += rhs[x + k];
        }

        offset_contribution0 *= lhs_offset;
        offset_contribution1 *= lhs_offset;

        offset_contribution0 += bias[n_idx];
        offset_contribution1 += bias[n_idx + 1];

        int32_t m_idx = 0;

        for (; m_idx <= (m - 2); m_idx+=2)
        {
            const q7_t *rhs_ptr = &rhs[0];

            // Initialize the accumulators with the offset contribution
            int32_t res00 = offset_contribution0;
            int32_t res01 = offset_contribution1;
            int32_t res10 = offset_contribution0;
            int32_t res11 = offset_contribution1;

            int32_t k_idx = 0;
            for (; k_idx <= (k - 16); k_idx+=16)
            {
                // Load 4 input values from the LHS/RHS matrix
                uint32_t rhs00 = *((uint32_t*)&rhs_ptr[0]);
                rhs_ptr += 4;
                uint32_t rhs10 = *((uint32_t*)&rhs_ptr[off0]);
                uint32_t lhs00 = *((uint32_t*)&lhs_ptr[0]);
                lhs_ptr += 4;

                // SMALD performs the multiply accumulate for two 16-bit input values
                // In order to use it, we need to extend two 8-bit values to 16-bit values
                // Since we load four 8-bit input values, we need two registers to hold the extended 16-bit values
                // sxtb16 extracts the bits[23:16] and bits[7:0]
                // sxtb16, ROR #8 rotate by 8 bits the input register and extracts the bits[23:16] and bits[7:0]
                uint32_t rhs01 = __SXTB16(rhs00);
                uint32_t lhs01 = __SXTB16(lhs00);
                uint32_t rhs11 = __SXTB16(rhs10);
                rhs00 = __SXTB16_ROR8(rhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                rhs10 = __SXTB16_ROR8(rhs10);

                res00 = __SMLAD(lhs00, rhs00, res00);
                res00 = __SMLAD(lhs01, rhs01, res00);
                res01 = __SMLAD(lhs00, rhs10, res01);
                res01 = __SMLAD(lhs01, rhs11, res01);

                lhs00 = *((uint32_t*)&lhs_ptr[off0]);
                lhs01 = __SXTB16(lhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                res10 = __SMLAD(lhs00, rhs00, res10);
                res10 = __SMLAD(lhs01, rhs01, res10);
                res11 = __SMLAD(lhs00, rhs10, res11);
                res11 = __SMLAD(lhs01, rhs11, res11);

                rhs00 = *((uint32_t*)&rhs_ptr[0]);
                rhs_ptr += 4;
                rhs10 = *((uint32_t*)&rhs_ptr[off0]);
                lhs00 = *((uint32_t*)&lhs_ptr[0]);
                lhs_ptr += 4;

                rhs01 = __SXTB16(rhs00);
                lhs01 = __SXTB16(lhs00);
                rhs11 = __SXTB16(rhs10);
                rhs00 = __SXTB16_ROR8(rhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                rhs10 = __SXTB16_ROR8(rhs10);

                res00 = __SMLAD(lhs00, rhs00, res00);
                res00 = __SMLAD(lhs01, rhs01, res00);
                res01 = __SMLAD(lhs00, rhs10, res01);
                res01 = __SMLAD(lhs01, rhs11, res01);

                lhs00 = *((uint32_t*)&lhs_ptr[off0]);
                lhs01 = __SXTB16(lhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                res10 = __SMLAD(lhs00, rhs00, res10);
                res10 = __SMLAD(lhs01, rhs01, res10);
                res11 = __SMLAD(lhs00, rhs10, res11);
                res11 = __SMLAD(lhs01, rhs11, res11);

                rhs00 = *((uint32_t*)&rhs_ptr[0]);
                rhs_ptr += 4;
                rhs10 = *((uint32_t*)&rhs_ptr[off0]);
                lhs00 = *((uint32_t*)&lhs_ptr[0]);
                lhs_ptr += 4;

                rhs01 = __SXTB16(rhs00);
                lhs01 = __SXTB16(lhs00);
                rhs11 = __SXTB16(rhs10);
                rhs00 = __SXTB16_ROR8(rhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                rhs10 = __SXTB16_ROR8(rhs10);

                res00 = __SMLAD(lhs00, rhs00, res00);
                res00 = __SMLAD(lhs01, rhs01, res00);
                res01 = __SMLAD(lhs00, rhs10, res01);
                res01 = __SMLAD(lhs01, rhs11, res01);

                lhs00 = *((uint32_t*)&lhs_ptr[off0]);
                lhs01 = __SXTB16(lhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                res10 = __SMLAD(lhs00, rhs00, res10);
                res10 = __SMLAD(lhs01, rhs01, res10);
                res11 = __SMLAD(lhs00, rhs10, res11);
                res11 = __SMLAD(lhs01, rhs11, res11);

                rhs00 = *((uint32_t*)&rhs_ptr[0]);
                rhs_ptr += 4;
                rhs10 = *((uint32_t*)&rhs_ptr[off0]);
                lhs00 = *((uint32_t*)&lhs_ptr[0]);
                lhs_ptr += 4;

                rhs01 = __SXTB16(rhs00);
                lhs01 = __SXTB16(lhs00);
                rhs11 = __SXTB16(rhs10);
                rhs00 = __SXTB16_ROR8(rhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                rhs10 = __SXTB16_ROR8(rhs10);

                res00 = __SMLAD(lhs00, rhs00, res00);
                res00 = __SMLAD(lhs01, rhs01, res00);
                res01 = __SMLAD(lhs00, rhs10, res01);
                res01 = __SMLAD(lhs01, rhs11, res01);

                lhs00 = *((uint32_t*)&lhs_ptr[off0]);
                lhs01 = __SXTB16(lhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                res10 = __SMLAD(lhs00, rhs00, res10);
                res10 = __SMLAD(lhs01, rhs01, res10);
                res11 = __SMLAD(lhs00, rhs10, res11);
                res11 = __SMLAD(lhs01, rhs11, res11);
            }

            // Left-over accumulations
            for (; k_idx < k; ++k_idx)
            {
                uint32_t rhs_value0 = rhs_ptr[0];
                uint32_t rhs_value1 = rhs_ptr[k];
                uint32_t lhs_value  = lhs_ptr[0];

                res00 = __SMLAD(lhs_value, rhs_value0, res00);
                res01 = __SMLAD(lhs_value, rhs_value1, res01);

                lhs_value  = lhs_ptr[k];
                res10 = __SMLAD(lhs_value, rhs_value0, res10);
                res11 = __SMLAD(lhs_value, rhs_value1, res11);

                ++rhs_ptr;
                ++lhs_ptr;
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[n_idx], dst_shifts[n_idx]);
            res01 = arm_nn_requantize(res01, dst_multipliers[n_idx + 1], dst_shifts[n_idx + 1]);
            res10 = arm_nn_requantize(res10, dst_multipliers[n_idx], dst_shifts[n_idx]);
            res11 = arm_nn_requantize(res11, dst_multipliers[n_idx + 1], dst_shifts[n_idx + 1]);

            // Add offset
            res00 += dst_offset;
            res01 += dst_offset;
            res10 += dst_offset;
            res11 += dst_offset;

            // Clamp the result
            res00 = MAX(res00, activation_min);
            res00 = MIN(res00, activation_max);
            res01 = MAX(res01, activation_min);
            res01 = MIN(res01, activation_max);
            res10 = MAX(res10, activation_min);
            res10 = MIN(res10, activation_max);
            res11 = MAX(res11, activation_min);
            res11 = MIN(res11, activation_max);

            dst_ptr[0] = (q7_t)res00;
            dst_ptr[1] = (q7_t)res01;
            dst_ptr += n;
            dst_ptr[0] = (q7_t)res10;
            dst_ptr[1] = (q7_t)res11;
            dst_ptr += n;

            lhs_ptr += k;
        }

        // Left-over rows
        for (; m_idx < m; ++m_idx)
        {
            const q7_t *rhs_ptr = &rhs[0];

            // Initialize the accumulators with the offset contribution
            int32_t res00 = offset_contribution0;
            int32_t res01 = offset_contribution1;

            int32_t k_idx = 0;
            for (; k_idx <= (k - 16); k_idx+=16)
            {
                // Load 4 input values from the LHS/RHS matrix
                uint32_t rhs00 = *((uint32_t*)&rhs_ptr[0]);
                rhs_ptr += 4;
                uint32_t rhs10 = *((uint32_t*)&rhs_ptr[off0]);
                uint32_t lhs00 = *((uint32_t*)&lhs_ptr[0]);
                lhs_ptr += 4;

                // SMALD performs the multiply accumulate for two 16-bit input values
                // In order to use it, we need to extend two 8-bit values to 16-bit values
                // Since we load four 8-bit input values, we need two registers to hold the extended 16-bit values
                // sxtb16 extracts the bits[23:16] and bits[7:0]
                // sxtb16, ROR #8 rotate by 8 bits the input register and extracts the bits[23:16] and bits[7:0]
                uint32_t rhs01 = __SXTB16(rhs00);
                uint32_t lhs01 = __SXTB16(lhs00);
                uint32_t rhs11 = __SXTB16(rhs10);
                rhs00 = __SXTB16_ROR8(rhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                rhs10 = __SXTB16_ROR8(rhs10);

                res00 = __SMLAD(lhs00, rhs00, res00);
                res00 = __SMLAD(lhs01, rhs01, res00);
                res01 = __SMLAD(lhs00, rhs10, res01);
                res01 = __SMLAD(lhs01, rhs11, res01);

                rhs00 = *((uint32_t*)&rhs_ptr[0]);
                rhs_ptr += 4;
                rhs10 = *((uint32_t*)&rhs_ptr[off0]);
                lhs00 = *((uint32_t*)&lhs_ptr[0]);
                lhs_ptr += 4;

                rhs01 = __SXTB16(rhs00);
                lhs01 = __SXTB16(lhs00);
                rhs11 = __SXTB16(rhs10);
                rhs00 = __SXTB16_ROR8(rhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                rhs10 = __SXTB16_ROR8(rhs10);

                res00 = __SMLAD(lhs00, rhs00, res00);
                res00 = __SMLAD(lhs01, rhs01, res00);
                res01 = __SMLAD(lhs00, rhs10, res01);
                res01 = __SMLAD(lhs01, rhs11, res01);

                rhs00 = *((uint32_t*)&rhs_ptr[0]);
                rhs_ptr += 4;
                rhs10 = *((uint32_t*)&rhs_ptr[off0]);
                lhs00 = *((uint32_t*)&lhs_ptr[0]);
                lhs_ptr += 4;

                rhs01 = __SXTB16(rhs00);
                lhs01 = __SXTB16(lhs00);
                rhs11 = __SXTB16(rhs10);
                rhs00 = __SXTB16_ROR8(rhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                rhs10 = __SXTB16_ROR8(rhs10);

                res00 = __SMLAD(lhs00, rhs00, res00);
                res00 = __SMLAD(lhs01, rhs01, res00);
                res01 = __SMLAD(lhs00, rhs10, res01);
                res01 = __SMLAD(lhs01, rhs11, res01);

                rhs00 = *((uint32_t*)&rhs_ptr[0]);
                rhs_ptr += 4;
                rhs10 = *((uint32_t*)&rhs_ptr[off0]);
                lhs00 = *((uint32_t*)&lhs_ptr[0]);
                lhs_ptr += 4;

                rhs01 = __SXTB16(rhs00);
                lhs01 = __SXTB16(lhs00);
                rhs11 = __SXTB16(rhs10);
                rhs00 = __SXTB16_ROR8(rhs00);
                lhs00 = __SXTB16_ROR8(lhs00);
                rhs10 = __SXTB16_ROR8(rhs10);

                res00 = __SMLAD(lhs00, rhs00, res00);
                res00 = __SMLAD(lhs01, rhs01, res00);
                res01 = __SMLAD(lhs00, rhs10, res01);
                res01 = __SMLAD(lhs01, rhs11, res01);
            }

            // Left-over accumulations
            for (; k_idx < k; ++k_idx)
            {
                uint32_t rhs_value0 = rhs_ptr[0];
                uint32_t rhs_value1 = rhs_ptr[k];
                uint32_t lhs_value  = lhs_ptr[0];

                res00 = __SMLAD(lhs_value, rhs_value0, res00);
                res01 = __SMLAD(lhs_value, rhs_value1, res01);

                ++rhs_ptr;
                ++lhs_ptr;
            }

            // Quantize down
            res00 = arm_nn_requantize(res00, dst_multipliers[n_idx], dst_shifts[n_idx]);
            res01 = arm_nn_requantize(res01, dst_multipliers[n_idx + 1], dst_shifts[n_idx + 1]);

            // Add offset
            res00 += dst_offset;
            res01 += dst_offset;

            // Clamp the result
            res00 = MAX(res00, activation_min);
            res00 = MIN(res00, activation_max);
            res01 = MAX(res01, activation_min);
            res01 = MIN(res01, activation_max);

            dst_ptr[0] = (q7_t)res00;
            dst_ptr[1] = (q7_t)res01;
            dst_ptr += n;
        }

        rhs += 2 * k;
        dst += 2;
    }

    return dst;
#else // defined(ARM_MATH_LOOPUNROLL) && defined(ARM_MATH_DSP)
    // TODO: Add support for no-DSP extension
    (void)lhs;
    (void)rhs;
    (void)bias;
    (void)dst_multipliers;
    (void)dst_shifts;
    (void)m;
    (void)n;
    (void)k;
    (void)lhs_offset;
    (void)dst_offset;
    (void)activation_min;
    (void)activation_max;
    (void)dst;
    return NULL;
#endif
}