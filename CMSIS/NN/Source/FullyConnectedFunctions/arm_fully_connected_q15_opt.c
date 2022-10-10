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
 * Title:        arm_fully_connected_q15_opt.c
 * Description:  Q15 opt fully-connected layer function
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
 * @addtogroup FC
 * @{
 */

/*
 * @brief Q15 opt fully-connected layer function
 * Refer function header for details
 */

arm_cmsis_nn_status arm_fully_connected_q15_opt(const q15_t *pV,
                                                const q15_t *pM,
                                                const uint16_t dim_vec,
                                                const uint16_t num_of_rows,
                                                const uint16_t bias_shift,
                                                const uint16_t out_shift,
                                                const q15_t *bias,
                                                q15_t *pOut,
                                                q15_t *vec_buffer)
{
    (void)vec_buffer;
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    const q15_t *pB = pM;
    q15_t *pO = pOut;
    const q15_t *pBias = bias;
    const q15_t *pA = pV;

    uint16_t rowCnt = num_of_rows >> 2;

    while (rowCnt)
    {
        q31_t sum = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t sum2 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t sum3 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t sum4 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

        uint16_t colCnt = dim_vec >> 1;

        pA = pV;

#ifdef USE_INTRINSIC

        while (colCnt)
        {
            q31_t inM11, inM12, inM13, inM14;
            q31_t inV;

            inV = arm_nn_read_q15x2_ia(&pA);
            inM11 = arm_nn_read_q15x2_ia(&pB);
            sum = __SMLAD(inV, inM11, sum);
            inM12 = arm_nn_read_q15x2_ia(&pB);
            sum2 = __SMLAD(inV, inM12, sum2);
            inM13 = arm_nn_read_q15x2_ia(&pB);
            sum3 = __SMLAD(inV, inM13, sum3);
            inM14 = arm_nn_read_q15x2_ia(&pB);
            sum4 = __SMLAD(inV, inM14, sum4);
            colCnt--;
        }

#else

        /*
         * register needed:
         * loop counter: colCnt
         * accumulators: sum, sum2, sum3, sum4
         * pointers: pB, pA
         * weight data: inM11, inM12, inM13, inM14
         * activation data: inV
         */

        asm volatile(
            "COL_LOOP_%=:\n"
            "ldr.w r4, [%[pA]], #4\n"
            "ldr.w r0, [%[pB]], #16\n"
            "smlad %[sum], r4, r0, %[sum]\n"
            "ldr.w r1, [%[pB] , #-12]\n"
            "smlad %[sum2], r4, r1, %[sum2]\n"
            "ldr.w r2, [%[pB] , #-8]\n"
            "smlad %[sum3], r4, r2, %[sum3]\n"
            "ldr.w r3, [%[pB] , #-4]\n"
            "smlad %[sum4], r4, r3, %[sum4]\n"
            "subs %[colCnt], #1\n"
            "bne COL_LOOP_%=\n"
            : [sum] "+r"(sum), [sum2] "+r"(sum2), [sum3] "+r"(sum3), [sum4] "+r"(sum4), [pB] "+r"(pB), [pA] "+r"(pA)
            : [colCnt] "r"(colCnt)
            : "r0", "r1", "r2", "r3", "r4");

#endif /* USE_INTRINSIC */

        colCnt = dim_vec & 0x1;
        while (colCnt)
        {

            q15_t inV = *pA++;
            q15_t inM = *pB++;
            q15_t inM2 = *pB++;
            q15_t inM3 = *pB++;
            q15_t inM4 = *pB++;

            sum += inV * inM;
            sum2 += inV * inM2;
            sum3 += inV * inM3;
            sum4 += inV * inM4;
            colCnt--;
        } /* while over colCnt */
        *pO++ = (q15_t)(__SSAT((sum >> out_shift), 16));
        *pO++ = (q15_t)(__SSAT((sum2 >> out_shift), 16));
        *pO++ = (q15_t)(__SSAT((sum3 >> out_shift), 16));
        *pO++ = (q15_t)(__SSAT((sum4 >> out_shift), 16));

        /* adjust the pointers and counters */
        rowCnt--;
    }

    /* left-over part of the rows */
    rowCnt = num_of_rows & 0x3;

    while (rowCnt)
    {
        q31_t sum = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

        uint16_t colCnt = dim_vec >> 2;

        pA = pV;

        while (colCnt)
        {
            q31_t inV1, inV2, inM1, inM2;

            inM1 = arm_nn_read_q15x2_ia(&pB);
            inV1 = arm_nn_read_q15x2_ia(&pA);
            sum = __SMLAD(inV1, inM1, sum);

            inM2 = arm_nn_read_q15x2_ia(&pB);
            inV2 = arm_nn_read_q15x2_ia(&pA);
            sum = __SMLAD(inV2, inM2, sum);

            colCnt--;
        }

        /* left-over of the vector */
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q15_t inV = *pA++;
            q15_t inM = *pB++;
            sum += inV * inM;
            colCnt--;
        }

        *pO++ = (q15_t)(__SSAT((sum >> out_shift), 16));

        rowCnt--;
    }

#else
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    uint16_t rowCnt = num_of_rows >> 2;
    const q15_t *pB = pM;
    const q15_t *pA;
    q15_t *pO = pOut;
    const q15_t *pBias = bias;

    while (rowCnt)
    {
        q31_t sum = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t sum2 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t sum3 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        q31_t sum4 = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);

        uint16_t colCnt = dim_vec >> 1;

        pA = pV;
        while (colCnt)
        {
            q15_t inA1 = *pA++;
            q15_t inA2 = *pA++;

            q15_t inB1 = *pB++;
            q15_t inB2 = *pB++;
            sum += inA1 * inB1 + inA2 * inB2;

            inB1 = *pB++;
            inB2 = *pB++;
            sum2 += inA1 * inB1 + inA2 * inB2;

            inB1 = *pB++;
            inB2 = *pB++;
            sum3 += inA1 * inB1 + inA2 * inB2;

            inB1 = *pB++;
            inB2 = *pB++;
            sum4 += inA1 * inB1 + inA2 * inB2;

            colCnt--;
        }
        colCnt = dim_vec & 0x1;
        while (colCnt)
        {
            q15_t inA = *pA++;
            q15_t inB = *pB++;
            sum += inA * inB;
            inB = *pB++;
            sum2 += inA * inB;
            inB = *pB++;
            sum3 += inA * inB;
            inB = *pB++;
            sum4 += inA * inB;
            colCnt--;
        }
        *pO++ = (q15_t)__SSAT((sum >> out_shift), 16);
        *pO++ = (q15_t)__SSAT((sum2 >> out_shift), 16);
        *pO++ = (q15_t)__SSAT((sum3 >> out_shift), 16);
        *pO++ = (q15_t)__SSAT((sum4 >> out_shift), 16);

        rowCnt--;
    }
    rowCnt = num_of_rows & 0x3;

    while (rowCnt)
    {
        int ip_out = ((q31_t)(*pBias++) << bias_shift) + NN_ROUND(out_shift);
        int j;

        pA = pV;
        for (j = 0; j < dim_vec; j++)
        {
            q15_t inA = *pA++;
            q15_t inB = *pB++;
            ip_out += inA * inB;
        }
        *pO++ = (q15_t)__SSAT((ip_out >> out_shift), 16);

        rowCnt--;
    }

#endif /* ARM_MATH_DSP */

    /* Return to ARM_CMSIS_NN_SUCCESS */
    return (ARM_CMSIS_NN_SUCCESS);
}

/**
 * @} end of FC group
 */
