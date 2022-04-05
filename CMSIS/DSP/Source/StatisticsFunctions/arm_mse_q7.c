/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_mse_q7.c
 * Description:  Mean square error between two Q7 vectors
 *
 * $Date:        04 April 2022
 * $Revision:    V1.10.0
 *
 * Target Processor: Cortex-M and Cortex-A cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2022 ARM Limited or its affiliates. All rights reserved.
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

#include "dsp/statistics_functions.h"

/**
  @ingroup groupStats
 */

/**
  @defgroup mse Mean Square Error

  Calculates the mean square error between two vectors.

 */

/**
  @addtogroup mse
  @{
 */

/**
  @brief         Mean square error between two Q7 vectors.
  @param[in]     pSrcA       points to the first input vector
  @param[in]     pSrcB       points to the second input vector
  @param[in]     blockSize  number of samples in input vector
  @param[out]    pResult    mean square error
  @return        none

*/
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
void arm_mse_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
        uint32_t blockSize,
        q7_t * pResult)
{
    uint32_t  blkCnt;           /* loop counters */
    q7x16_t vecSrcA,vecSrcB;
    q31_t   sum = 0LL;
    q7_t inA,inB;

   /* Compute 16 outputs at a time */
    blkCnt = blockSize >> 4U;
    while (blkCnt > 0U)
    {
        vecSrcA = vldrbq_s8(pSrcA);
        vecSrcB = vldrbq_s8(pSrcB);

        vecSrcA = vqsubq(vecSrcA,vecSrcB);
        /*
         * sum lanes
         */
        sum = vmladavaq(sum, vecSrcA, vecSrcA);

        blkCnt--;
        pSrcA += 16;
        pSrcB += 16;
    }

    /*
     * tail
     */
    blkCnt = blockSize & 0xF;
    while (blkCnt > 0U)
    {
       /* C = A[0] * A[0] + A[1] * A[1] + ... + A[blockSize-1] * A[blockSize-1] */

       /* Compute Power and store result in a temporary variable, sum. */
       inA = *pSrcA++;
       inB = *pSrcB++;

       inA = (q7_t) __SSAT((q15_t) inA - (q15_t)inB, 8);

       sum += ((q15_t) inA * inA);

       /* Decrement loop counter */
       blkCnt--;
    }

    *pResult = (q7_t) __SSAT((q15_t) (sum / blockSize)>>7, 8);
}
#else
void arm_mse_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
        uint32_t blockSize,
        q7_t * pResult)
{
        uint32_t blkCnt;                               /* Loop counter */
        q31_t sum = 0;                                 /* Temporary result storage */
        q7_t inA,inB;                                       /* Temporary variable to store input value */

#if defined (ARM_MATH_LOOPUNROLL) && defined (ARM_MATH_DSP)
        q31_t inA32,inB32;                                    /* Temporary variable to store packed input value */
        q31_t in1, in2;                                /* Temporary variables to store input value */
#endif

#if defined (ARM_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A[0] * A[0] + A[1] * A[1] + ... + A[blockSize-1] * A[blockSize-1] */

    /* Compute Power and store result in a temporary variable, sum. */
#if defined (ARM_MATH_DSP)
    inA32 = read_q7x4_ia ((q7_t **) &pSrcA);
    inB32 = read_q7x4_ia ((q7_t **) &pSrcB);

    inA32 = __QSUB8(inA32, inB32);

    in1 = __SXTB16(__ROR(inA32, 8));
    in2 = __SXTB16(inA32);

    /* calculate power and accumulate to accumulator */
    sum = __SMLAD(in1, in1, sum);
    sum = __SMLAD(in2, in2, sum);
#else
    inA = *pSrcA++;
    inB = *pSrcB++;
    inA = (q7_t) __SSAT((q15_t) inA - (q15_t)inB, 8);
    sum += ((q15_t) inA * inA);

    inA = *pSrcA++;
    inB = *pSrcB++;
    inA = (q7_t) __SSAT((q15_t) inA - (q15_t)inB, 8);
    sum += ((q15_t) inA * inA);

    inA = *pSrcA++;
    inB = *pSrcB++;
    inA = (q7_t) __SSAT((q15_t) inA - (q15_t)inB, 8);
    sum += ((q15_t) inA * inA);

    inA = *pSrcA++;
    inB = *pSrcB++;
    inA = (q7_t) __SSAT((q15_t) inA - (q15_t)inB, 8);
    sum += ((q15_t) inA * inA);
#endif /* #if defined (ARM_MATH_DSP) */

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize % 0x4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* C = A[0] * A[0] + A[1] * A[1] + ... + A[blockSize-1] * A[blockSize-1] */

    /* Compute Power and store result in a temporary variable, sum. */
    inA = *pSrcA++;
    inB = *pSrcB++;

    inA = (q7_t) __SSAT((q15_t) inA - (q15_t)inB, 8);
    sum += ((q15_t) inA * inA);

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Store result in q7 format */
  *pResult = (q7_t) __SSAT((q15_t) (sum / blockSize)>>7, 8);;
}
#endif /* defined(ARM_MATH_MVEI) */

/**
  @} end of power group
 */
