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
 * Title:        arm_fully_connected_s8
 * Description:  Fully connected function compatible with TF Lite.
 *
 * $Date:        10. July 2019
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M and Cortex-A cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup FC
 * @{
 */

/*
   * S8 basic fully-connected and matrix multiplication layer function for TensorFlow Lite
   *
   * Refer header file for details.
   *
   */

#if defined(ARM_MATH_MVEI)

arm_status
arm_fully_connected_s8(const int8_t *input,
                       const int8_t *kernel,
                       const uint16_t col_dim,
                       const uint16_t row_dim,
                       const uint16_t nb_batches,
                       const int32_t input_offset,
                       const int32_t filter_offset,
                       const int32_t out_mult,
                       const int32_t out_shift,
                       const int32_t output_offset,
                       const int32_t *bias,
                       int8_t *output,
                       const int32_t output_activation_min,
                       const int32_t output_activation_max,
                       q15_t *vec_buffer)
{
    (void)vec_buffer;
    const int8_t *input_a;
    const int32_t *bias_tmp = bias;
    const int8_t *weight_tmp = kernel;
    int32_t batch_count = nb_batches;

    const int16x8_t filter_offset_vec = vdupq_n_s16((int16_t)filter_offset);
    const int16x8_t input_offset_vec = vdupq_n_s16((int16_t)input_offset);

    while (batch_count)
    {
        bias_tmp = bias;
        weight_tmp = kernel;

        int cnt;
        cnt = row_dim >> 2;

        for (int out_c = 0; out_c < cnt; out_c++)
        {
            int32_t acc1 = *bias_tmp++;
            int32_t acc2 = *bias_tmp++;
            int32_t acc3 = *bias_tmp++;
            int32_t acc4 = *bias_tmp++;
            input_a = input;

            int16x8_t input_val, filter_val;
            int16x8_t tmp_a1, tmp_a2, tmp_a3, tmp_a4, tmp_b;
            int32x4_t acc;
            int32_t block_count;

            const int8_t *col = input_a;
            const int8_t *row_0 = weight_tmp;
            const int8_t *row_1 = weight_tmp + col_dim;
            const int8_t *row_2 = weight_tmp + 2 * col_dim;
            const int8_t *row_3 = weight_tmp + 3 * col_dim;

            block_count = col_dim >> 3U;

            while (block_count > 0U)
            {
                input_val = vldrbq_s16(col);
                tmp_b = vaddq_s16(input_val, input_offset_vec);

                filter_val = vldrbq_s16(row_0);
                tmp_a1 = vaddq_s16(filter_val, filter_offset_vec);
                acc1 = vmladavaq_s16(acc1, tmp_a1, tmp_b);

                filter_val = vldrbq_s16(row_1);
                tmp_a2 = vaddq_s16(filter_val, filter_offset_vec);
                acc2 = vmladavaq_s16(acc2, tmp_a2, tmp_b);

                filter_val = vldrbq_s16(row_2);
                tmp_a3 = vaddq_s16(filter_val, filter_offset_vec);
                acc3 = vmladavaq_s16(acc3, tmp_a3, tmp_b);

                filter_val = vldrbq_s16(row_3);
                tmp_a4 = vaddq_s16(filter_val, filter_offset_vec);
                acc4 = vmladavaq_s16(acc4, tmp_a4, tmp_b);

                col += 8;
                row_0 += 8;
                row_1 += 8;
                row_2 += 8;
                row_3 += 8;
                block_count--;
            }

            block_count = col_dim & 7;

            while (block_count > 0U)
            {
                q15_t col_ip = *col++;

                q7_t in_m1 = *row_0++;
                q7_t in_m2 = *row_1++;
                q7_t in_m3 = *row_2++;
                q7_t in_m4 = *row_3++;

                acc1 += (col_ip + input_offset) * (in_m1 + filter_offset);
                acc2 += (col_ip + input_offset) * (in_m2 + filter_offset);
                acc3 += (col_ip + input_offset) * (in_m3 + filter_offset);
                acc4 += (col_ip + input_offset) * (in_m4 + filter_offset);

                block_count--;
            }

            input_a = input + col_dim;
            weight_tmp += 4 * col_dim;

            acc[0] = acc1;
            acc[1] = acc2;
            acc[2] = acc3;
            acc[3] = acc4;

            acc = arm_requantize_mve(acc, out_mult, out_shift);
            acc = vaddq_s32(acc, vdupq_n_s32(output_offset));
            acc = vmaxq_s32(acc, vdupq_n_s32(output_activation_min));
            acc = vminq_s32(acc, vdupq_n_s32(output_activation_max));

            vstrbq_s32(output, acc);

            output += 4;
        }

        cnt = row_dim & 3;
        for (int out_c = 0; out_c < cnt; out_c++)
        {

            int32_t acc = *bias_tmp++;
            input_a = input;

            int16x8_t input_val, filter_val;
            int16x8_t tmp_a, tmp_b;
            int32x4_t vecSum;
            int32_t block_count;

            const int8_t *col = input_a;
            const int8_t *kernel_cur = weight_tmp;

            block_count = col_dim >> 3U;

            while (block_count > 0U)
            {

                input_val = vldrbq_s16(col);
                filter_val = vldrbq_s16(kernel_cur);

                tmp_a = vaddq_s16(filter_val, filter_offset_vec);
                tmp_b = vaddq_s16(input_val, input_offset_vec);

                acc = vmladavaq_s16(acc, tmp_a, tmp_b);

                col += 8;
                kernel_cur += 8;
                block_count--;
            }

            block_count = col_dim & 7;

            while (block_count > 0U)
            {
                q15_t col_ip = *col++;
                q7_t in_m = *kernel_cur++;

                acc += (col_ip + input_offset) * (in_m + filter_offset);

                block_count--;
            }

            input_a += col_dim;
            weight_tmp += col_dim;

            acc = arm_nn_sat_doubling_high_mult(acc * (1 << LEFT_SHIFT(out_shift)), out_mult);
            acc = arm_nn_divide_by_power_of_two(acc, RIGHT_SHIFT(out_shift));

            acc += output_offset;

            acc = MAX(acc, output_activation_min);
            acc = MIN(acc, output_activation_max);

            *output++ = (int8_t)(acc);
        }
        input += col_dim;
        batch_count--;
    }
    return (ARM_MATH_SUCCESS);
}

#else
arm_status
arm_fully_connected_s8(const int8_t *input,
                       const int8_t *kernel,
                       const uint16_t col_dim,
                       const uint16_t row_dim,
                       const uint16_t nb_batches,
                       const int32_t input_offset,
                       const int32_t filter_offset,
                       const int32_t out_mult,
                       const int32_t out_shift,
                       const int32_t output_offset,
                       const int32_t *bias,
                       int8_t *output,
                       const int32_t output_activation_min,
                       const int32_t output_activation_max,
                       q15_t *vec_buffer)
{
#if defined(ARM_MATH_LOOPUNROLL) && defined(ARM_MATH_DSP)
    q31_t acc;

    uint16_t batchCnt = nb_batches;

    /* CMSIS-DSP and NN are generally using q7 and q15 types.
     Here we are computing with s8 and not q7.
     So, q7_t is not really the right type to use but
     it is kept for consistency with some function APIs
     which are used in this implementation.
     */
    const int32_t *pBiasTmp = bias;
    const q7_t *pB = kernel;
    const q7_t *pB2;
    q7_t *pO = output;
    q15_t *pA;
    q31_t ioffset;
    q31_t foffset;

    ioffset = ((input_offset & 0x0FFFF) << 16) | (input_offset & 0x0FFFF);
    foffset = ((filter_offset & 0x0FFFF) << 16) | (filter_offset & 0x0FFFF);

    while (batchCnt)
    {
        pBiasTmp = bias;
        pB = kernel;
        arm_q7_to_q15_reordered_no_shift(input, vec_buffer, col_dim);
        uint16_t rowCnt = row_dim >> 1;
        /* Unroll on the rows */
        while (rowCnt)
        {
            q31_t sum = (q31_t)(*pBiasTmp++);
            q31_t sum2 = (q31_t)(*pBiasTmp++);
            uint16_t colCnt = col_dim >> 2;

            pA = vec_buffer;
            pB2 = pB + col_dim;

            /* Vectorize on the columns */
            while (colCnt)
            {
                q31_t inV, inM11, inM12, inM21, inM22;
                pB = read_and_pad_reordered_with_offset(pB, &inM11, &inM12, foffset);
                pB2 = read_and_pad_reordered_with_offset(pB2, &inM21, &inM22, foffset);

                inV = read_q15x2_ia(&pA);
                inV = __QADD16(inV, ioffset);

                sum = __SMLAD(inV, inM11, sum);
                sum2 = __SMLAD(inV, inM21, sum2);

                inV = read_q15x2_ia(&pA);
                inV = __QADD16(inV, ioffset);

                sum = __SMLAD(inV, inM12, sum);
                sum2 = __SMLAD(inV, inM22, sum2);

                colCnt--;
            }

            /* Column vector tail */
            colCnt = col_dim & 0x3;
            while (colCnt)
            {
                q15_t inV = *pA++;
                q7_t inM = *pB++;
                q7_t inM2 = *pB2++;

                sum += (inV + input_offset) * (inM + filter_offset);
                sum2 += (inV + input_offset) * (inM2 + filter_offset);
                colCnt--;
            }

            acc = arm_nn_sat_doubling_high_mult(sum * (1 << LEFT_SHIFT(out_shift)), out_mult);
            acc = arm_nn_divide_by_power_of_two(acc, RIGHT_SHIFT(out_shift));
            acc += output_offset;
            acc = MAX(acc, output_activation_min);
            acc = MIN(acc, output_activation_max);

            *pO++ = (q7_t)(acc);

            acc = arm_nn_sat_doubling_high_mult(sum2 * (1 << LEFT_SHIFT(out_shift)), out_mult);
            acc = arm_nn_divide_by_power_of_two(acc, RIGHT_SHIFT(out_shift));
            acc += output_offset;
            acc = MAX(acc, output_activation_min);
            acc = MIN(acc, output_activation_max);
            *pO++ = (q7_t)(acc);

            pB += col_dim;
            rowCnt--;
        }

        /* left-over part of the rows */
        rowCnt = row_dim & 0x1;

        while (rowCnt)
        {
            uint16_t colCnt = col_dim >> 2;
            q31_t sum = (q31_t)(*pBiasTmp++);

            pA = vec_buffer;

            /* Vectorize on the columns */
            while (colCnt)
            {
                q31_t inV, inM11, inM12;

                pB = read_and_pad_reordered_with_offset(pB, &inM11, &inM12, foffset);

                inV = read_q15x2_ia(&pA);
                inV = __QADD16(inV, ioffset);

                sum = __SMLAD(inV, inM11, sum);

                inV = read_q15x2_ia(&pA);
                inV = __QADD16(inV, ioffset);

                sum = __SMLAD(inV, inM12, sum);

                colCnt--;
            }

            /* Column vector tail */
            colCnt = col_dim & 0x3;

            while (colCnt)
            {
                q15_t inV = *pA++;
                q7_t inM = *pB++;
                sum += (inV + input_offset) * (inM + filter_offset);
                colCnt--;
            }

            acc = arm_nn_sat_doubling_high_mult(sum * (1 << LEFT_SHIFT(out_shift)), out_mult);
            acc = arm_nn_divide_by_power_of_two(acc, RIGHT_SHIFT(out_shift));
            acc += output_offset;
            acc = MAX(acc, output_activation_min);
            acc = MIN(acc, output_activation_max);
            *pO++ = (q7_t)(acc);

            rowCnt--;
        }
        input += col_dim;
        batchCnt--;
    }
    return (ARM_MATH_SUCCESS);

#else
    (void)vec_buffer;
    const int8_t *pInputA;
    const int32_t *pBiasTmp = bias;
    const int8_t *pWeightTmp = kernel;
    uint16_t batchCnt = nb_batches;

    while (batchCnt)
    {
        pBiasTmp = bias;
        pWeightTmp = kernel;
        for (int out_c = 0; out_c < row_dim; out_c++)
        {

            int32_t acc = *pBiasTmp++;
            pInputA = input;
            for (int d = 0; d < col_dim; d++)
            {

                int32_t input_val = *pInputA++;

                int32_t filter_val = *pWeightTmp++;

                acc += (filter_val + filter_offset) * (input_val + input_offset);
            }

            acc = arm_nn_sat_doubling_high_mult(acc * (1 << LEFT_SHIFT(out_shift)), out_mult);
            acc = arm_nn_divide_by_power_of_two(acc, RIGHT_SHIFT(out_shift));

            acc += output_offset;

            acc = MAX(acc, output_activation_min);
            acc = MIN(acc, output_activation_max);

            *output++ = (int8_t)(acc);
        }
        input += col_dim;
        batchCnt--;
    }
    return (ARM_MATH_SUCCESS);
#endif /*  defined(ARM_MATH_LOOPUNROLL) && defined (ARM_MATH_DSP) */
}
#endif /* ARM_MATH_HELIUM */

int32_t arm_fully_connected_s8_get_buffer_size(const uint16_t col_dim)
{
#if defined(ARM_MATH_LOOPUNROLL) && defined(ARM_MATH_DSP)
    return col_dim * sizeof(int16_t);
#else
    (void)col_dim;
    return 0;
#endif
}

/**
 * @} end of FC group
 */
