/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_dot_prod_q31.c
 * Description:  Q31 dot product
 *
 * $Date:        18. March 2019
 * $Revision:    V1.6.0
 *
 * Target Processor: Cortex-M cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2019 ARM Limited or its affiliates. All rights reserved.
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

#include "arm_math.h"

/**
  @ingroup groupMath
 */

/**
  @addtogroup BasicDotProd
  @{
 */

/**
  @brief         Dot product of Q31 vectors.
  @param[in]     pSrcA      points to the first input vector.
  @param[in]     pSrcB      points to the second input vector.
  @param[in]     blockSize  number of samples in each vector.
  @param[out]    result     output result returned here.
  @return        none

  @par           Scaling and Overflow Behavior
                   The intermediate multiplications are in 1.31 x 1.31 = 2.62 format and these
                   are truncated to 2.48 format by discarding the lower 14 bits.
                   The 2.48 result is then added without saturation to a 64-bit accumulator in 16.48 format.
                   There are 15 guard bits in the accumulator and there is no risk of overflow as long as
                   the length of the vectors is less than 2^16 elements.
                   The return result is in 16.48 format.
 */

#if defined(ARM_MATH_MVEI)

#include "arm_helium_utils.h"

void arm_dot_prod_q31(
    const q31_t * pSrcA,
    const q31_t * pSrcB,
    uint32_t blockSize,
    q63_t * result)
{
    uint32_t  blkCnt;           /* loop counters */
    q31x4_t vecA;
    q31x4_t vecB;
    q63_t     sum = 0LL;

    /* Compute 4 outputs at a time */
    blkCnt = blockSize >> 2;
    while (blkCnt > 0U)
    {
        /*
         * C = A[0]* B[0] + A[1]* B[1] + A[2]* B[2] + .....+ A[blockSize-1]* B[blockSize-1]
         * Calculate dot product and then store the result in a temporary buffer.
         */
        vecA = vld1q(pSrcA);
        vecB = vld1q(pSrcB);
        sum = vrmlaldavhaq(sum, vecA, vecB);
        /*
         * Decrement the blockSize loop counter
         */
        blkCnt--;
        /*
         * advance vector source and destination pointers
         */
        pSrcA += 4;
        pSrcB += 4;
    }
    /*
     * tail
     */
    blkCnt = blockSize & 3;
    if (blkCnt > 0U)
    {
        mve_pred16_t p0 = vctp32q(blkCnt);
        vecA = vld1q(pSrcA);
        vecB = vld1q(pSrcB);
        sum = vrmlaldavhaq_p(sum, vecA, vecB, p0);
    }

    /*
     * vrmlaldavhaq provides extra intermediate accumulator headroom.
     * limiting the need of intermediate scaling
     * Scalar variant uses 2.48 accu format by right shifting accumulators by 14.
     * 16.48 output conversion is performed outside the loop by scaling accu. by 6
     */
    *result = asrl(sum, (14 - 8));
}

#else
void arm_dot_prod_q31(
  const q31_t * pSrcA,
  const q31_t * pSrcB,
        uint32_t blockSize,
        q63_t * result)
{
        uint32_t blkCnt;                               /* Loop counter */
        q63_t sum = 0;                                 /* Temporary return variable */

#if defined(ARM_MATH_NEON_EXPERIMENTAL)
    int32x4_t vec1;
    int32x4_t vec2;
    int32x4_t res;
    int64x2_t accum1=vdupq_n_s64(0);
    int64x2_t accum2=vdupq_n_s64(0);

    /* Compute 4 outputs at a time */  
    blkCnt = blockSize >> 2U;

    vec1 = vld1q_s32(pSrcA);
    vec2 = vld1q_s32(pSrcB);

    while (blkCnt > 0U)
    {
        /* C = A[0]* B[0] + A[1]* B[1] + A[2]* B[2] + .....+ A[blockSize-1]* B[blockSize-1] */
        /* Calculate dot product and then store the result in a temporary buffer. */

        /* To be bit exact with original version, an intermediate shift of 14 is required
           after each multiplication.
           It is less efficient than a solution using multiply and
           accumulate instructions */
        accum1=vmull_s32(vget_low_s32(vec1), vget_low_s32(vec2));
        accum1=vshrq_n_s64(accum1,14);
        
        accum2=vmull_s32(vget_high_s32(vec1), vget_high_s32(vec2));
        accum2=vshrq_n_s64(accum2,14);
        accum2=vaddq_s64(accum2,accum1);

        sum += accum2[0] + accum2[1] ;

        /* Increment pointers */
        pSrcA += 4;
        pSrcB += 4; 

        vec1 = vld1q_s32(pSrcA);
        vec2 = vld1q_s32(pSrcB);

        
        /* Decrement the blockSize loop counter */
        blkCnt--;
    }

    /* Tail */
    blkCnt = blockSize & 0x3;

#else
#if defined (ARM_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A[0]* B[0] + A[1]* B[1] + A[2]* B[2] + .....+ A[blockSize-1]* B[blockSize-1] */

    /* Calculate dot product and store result in a temporary buffer. */
    sum += ((q63_t) *pSrcA++ * *pSrcB++) >> 14U;

    sum += ((q63_t) *pSrcA++ * *pSrcB++) >> 14U;

    sum += ((q63_t) *pSrcA++ * *pSrcB++) >> 14U;

    sum += ((q63_t) *pSrcA++ * *pSrcB++) >> 14U;

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize % 0x4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */
#endif /* #if defined (ARM_MATH_NEON) */

  while (blkCnt > 0U)
  {
    /* C = A[0]* B[0] + A[1]* B[1] + A[2]* B[2] + .....+ A[blockSize-1]* B[blockSize-1] */

    /* Calculate dot product and store result in a temporary buffer. */
    sum += ((q63_t) *pSrcA++ * *pSrcB++) >> 14U;

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Store result in destination buffer in 16.48 format */
  *result = sum;
}
#endif /* #if defined (ARM_MATH_MVEI) */

/**
  @} end of BasicDotProd group
 */
