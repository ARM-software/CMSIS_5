/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_sub_q7.c
 * Description:  Q7 vector subtraction
 *
 * $Date:        23 April 2021
 * $Revision:    V1.9.0
 *
 * Target Processor: Cortex-M and Cortex-A cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
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

#include "dsp/basic_math_functions.h"

/**
  @ingroup groupMath
 */

/**
  @addtogroup BasicSub
  @{
 */

/**
  @brief         Q7 vector subtraction.
  @param[in]     pSrcA      points to the first input vector
  @param[in]     pSrcB      points to the second input vector
  @param[out]    pDst       points to the output vector
  @param[in]     blockSize  number of samples in each vector
  @return        none

  @par           Scaling and Overflow Behavior
                   The function uses saturating arithmetic.
                   Results outside of the allowable Q7 range [0x80 0x7F] will be saturated.
 */
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)

#include "arm_helium_utils.h"

void arm_sub_q7(
    const q7_t * pSrcA,
    const q7_t * pSrcB,
    q7_t * pDst,
    uint32_t blockSize)
{
    uint32_t  blkCnt;           /* loop counters */
    q7x16_t vecA;
    q7x16_t vecB;

    /* Compute 16 outputs at a time */
    blkCnt = blockSize >> 4;
    while (blkCnt > 0U)
    {
        /*
         * C = A - B
         * Subtract and then store the results in the destination buffer.
         */
        vecA = vld1q(pSrcA);
        vecB = vld1q(pSrcB);
        vst1q(pDst, vqsubq(vecA, vecB));
        /*
         * Decrement the blockSize loop counter
         */
        blkCnt--;
        /*
         * advance vector source and destination pointers
         */
        pSrcA  += 16;
        pSrcB  += 16;
        pDst   += 16;
    }
    /*
     * tail
     */
    blkCnt = blockSize & 0xF;
    if (blkCnt > 0U)
    {
        mve_pred16_t p0 = vctp8q(blkCnt);
        vecA = vld1q(pSrcA);
        vecB = vld1q(pSrcB);
        vstrbq_p(pDst, vqsubq(vecA, vecB), p0);
    }
}
#else
void arm_sub_q7(
  const q7_t * pSrcA,
  const q7_t * pSrcB,
        q7_t * pDst,
        uint32_t blockSize)
{
        uint32_t blkCnt;                               /* Loop counter */

#if defined (ARM_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A - B */

#if defined (ARM_MATH_DSP)
    /* Subtract and store result in destination buffer (4 samples at a time). */
    write_q7x4_ia (&pDst, __QSUB8(read_q7x4_ia (&pSrcA), read_q7x4_ia (&pSrcB)));
#else
    *pDst++ = (q7_t) __SSAT((q15_t) *pSrcA++ - *pSrcB++, 8);
    *pDst++ = (q7_t) __SSAT((q15_t) *pSrcA++ - *pSrcB++, 8);
    *pDst++ = (q7_t) __SSAT((q15_t) *pSrcA++ - *pSrcB++, 8);
    *pDst++ = (q7_t) __SSAT((q15_t) *pSrcA++ - *pSrcB++, 8);
#endif

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
    /* C = A - B */

    /* Subtract and store result in destination buffer. */
    *pDst++ = (q7_t) __SSAT((q15_t) *pSrcA++ - *pSrcB++, 8);

    /* Decrement loop counter */
    blkCnt--;
  }

}
#endif /* defined(ARM_MATH_MVEI) */

/**
  @} end of BasicSub group
 */
