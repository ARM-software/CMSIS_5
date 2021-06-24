/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_abs_q7.c
 * Description:  Q7 vector absolute value
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
  @addtogroup BasicAbs
  @{
 */

/**
  @brief         Q7 vector absolute value.
  @param[in]     pSrc       points to the input vector
  @param[out]    pDst       points to the output vector
  @param[in]     blockSize  number of samples in each vector
  @return        none

  @par           Conditions for optimum performance
                   Input and output buffers should be aligned by 32-bit
  @par           Scaling and Overflow Behavior
                   The function uses saturating arithmetic.
                   The Q7 value -1 (0x80) will be saturated to the maximum allowable positive value 0x7F.
 */

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)

#include "arm_helium_utils.h"

void arm_abs_q7(
    const q7_t * pSrc,
    q7_t * pDst,
    uint32_t blockSize)
{
    uint32_t  blkCnt;           /* loop counters */
    q7x16_t vecSrc;

    /* Compute 16 outputs at a time */
    blkCnt = blockSize >> 4;
    while (blkCnt > 0U)
    {
        /*
         * C = |A|
         * Calculate absolute and then store the results in the destination buffer.
         */
        vecSrc = vld1q(pSrc);
        vst1q(pDst, vqabsq(vecSrc));
        /*
         * Decrement the blockSize loop counter
         */
        blkCnt--;
        /*
         * advance vector source and destination pointers
         */
        pSrc += 16;
        pDst += 16;
    }
    /*
     * tail
     */
    blkCnt = blockSize & 0xF;
    if (blkCnt > 0U)
    {
        mve_pred16_t p0 = vctp8q(blkCnt);
        vecSrc = vld1q(pSrc);
        vstrbq_p(pDst, vqabsq(vecSrc), p0);
    }
}

#else
void arm_abs_q7(
  const q7_t * pSrc,
        q7_t * pDst,
        uint32_t blockSize)
{
        uint32_t blkCnt;                               /* Loop counter */
        q7_t in;                                       /* Temporary input variable */

#if defined (ARM_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = |A| */

    /* Calculate absolute of input (if -1 then saturated to 0x7f) and store result in destination buffer. */
    in = *pSrc++;
#if defined (ARM_MATH_DSP)
    *pDst++ = (in > 0) ? in : (q7_t)__QSUB8(0, in);
#else
    *pDst++ = (in > 0) ? in : ((in == (q7_t) 0x80) ? (q7_t) 0x7f : -in);
#endif

    in = *pSrc++;
#if defined (ARM_MATH_DSP)
    *pDst++ = (in > 0) ? in : (q7_t)__QSUB8(0, in);
#else
    *pDst++ = (in > 0) ? in : ((in == (q7_t) 0x80) ? (q7_t) 0x7f : -in);
#endif

    in = *pSrc++;
#if defined (ARM_MATH_DSP)
    *pDst++ = (in > 0) ? in : (q7_t)__QSUB8(0, in);
#else
    *pDst++ = (in > 0) ? in : ((in == (q7_t) 0x80) ? (q7_t) 0x7f : -in);
#endif

    in = *pSrc++;
#if defined (ARM_MATH_DSP)
    *pDst++ = (in > 0) ? in : (q7_t)__QSUB8(0, in);
#else
    *pDst++ = (in > 0) ? in : ((in == (q7_t) 0x80) ? (q7_t) 0x7f : -in);
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
    /* C = |A| */

    /* Calculate absolute of input (if -1 then saturated to 0x7f) and store result in destination buffer. */
    in = *pSrc++;
#if defined (ARM_MATH_DSP)
    *pDst++ = (in > 0) ? in : (q7_t) __QSUB8(0, in);
#else
    *pDst++ = (in > 0) ? in : ((in == (q7_t) 0x80) ? (q7_t) 0x7f : -in);
#endif

    /* Decrement loop counter */
    blkCnt--;
  }

}
#endif /* defined(ARM_MATH_MVEI) */

/**
  @} end of BasicAbs group
 */
