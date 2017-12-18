/*
 * Copyright (C) 2010-2017 ARM Limited or its affiliates. All rights reserved.
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
 * Project:      CMSIS-NN
 * Title:        arm_fully_connected_q15.c
 * Description:  Q15 basic fully-connected layer function
 *
 * Target Processor: Cortex-M4 and Cortex-M7 cores
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

  /**
   * @brief Q15 opt fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * vec_buffer size: 0
   *
   */

arm_status
arm_fully_connected_q15(const q15_t * pV,
                        const q15_t * pM,
                        const uint16_t dim_vec,
                        const uint16_t num_of_rows,
                        const uint16_t bias_shift,
                        const uint16_t out_shift, const q15_t * bias, q15_t * pOut, q15_t * vec_buffer)
{

#if defined (ARM_MATH_DSP)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    uint16_t  i_row;
    const q15_t *pB = pM;
    q15_t    *pO = pOut;
    q15_t    *pA;

    /* this loop loops over different output */
    for (i_row = 0; i_row < num_of_rows; i_row++)
    {

        q31_t     sum = bias[i_row] << bias_shift;

        uint16_t  colCnt = dim_vec >> 2;

        pA = vec_buffer;

        while (colCnt)
        {
            q31_t     inV1, inV2, inM1, inM2;
            inV1 = *__SIMD32(pA)++;
            inM1 = *__SIMD32(pB)++;
            sum = __SMLAD(inV1, inM1, sum);

            inV2 = *__SIMD32(pA)++;
            inM2 = *__SIMD32(pB)++;

            sum = __SMLAD(inV2, inM2, sum);

            colCnt--;
        }
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q15_t     inV = *pA++;
            q15_t     inM = *pB++;

            sum += inV * inM;
            colCnt--;
        }                       /* while over colCnt */
        *pO++ = (q15_t) (__SSAT((sum >> out_shift), 16));
    }

#else
    int       i, j;
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    for (i = 0; i < num_of_rows; i++)
    {
        int       ip_out = bias[i] << bias_shift;
        for (j = 0; j < dim_vec; j++)
        {
            ip_out += pV[j] * pM[i * dim_vec + j];
        }
        pOut[i] = (q15_t) __SSAT((ip_out >> out_shift), 16);
    }

#endif                          /* ARM_MATH_DSP */

    /* Return to application */
    return (ARM_MATH_SUCCESS);

}

/**
 * @} end of FC group
 */
