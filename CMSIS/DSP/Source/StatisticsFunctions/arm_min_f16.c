/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_min_f16.c
 * Description:  Minimum value of a floating-point vector
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

#include "dsp/statistics_functions_f16.h"

#if defined(ARM_FLOAT16_SUPPORTED)


#if (defined(ARM_MATH_NEON) || defined(ARM_MATH_MVEF)) && !defined(ARM_MATH_AUTOVECTORIZE)
#include <limits.h>
#endif


/**
  @ingroup groupStats
 */

/**
  @addtogroup Min
  @{
 */

/**
  @brief         Minimum value of a floating-point vector.
  @param[in]     pSrc       points to the input vector
  @param[in]     blockSize  number of samples in input vector
  @param[out]    pResult    minimum value returned here
  @param[out]    pIndex     index of minimum value returned here
  @return        none
 */

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)

void arm_min_f16(
  const float16_t * pSrc,
  uint32_t blockSize,
  float16_t * pResult,
  uint32_t * pIndex)
{
    int32_t  blkCnt;           /* loop counters */
    f16x8_t vecSrc;
    float16_t const *pSrcVec;
    f16x8_t curExtremValVec = vdupq_n_f16(F16_MAX);
    float16_t minValue = F16_MAX;
    uint32_t  idx = blockSize;
    uint16x8_t indexVec;
    uint16x8_t curExtremIdxVec;
    mve_pred16_t p0;

    indexVec = vidupq_u16((uint32_t)0, 1);
    curExtremIdxVec = vdupq_n_u16(0);

    pSrcVec = (float16_t const *) pSrc;
    blkCnt = blockSize >> 3;
    while (blkCnt > 0)
    {
        vecSrc = vldrhq_f16(pSrcVec);  pSrcVec += 8;
        /*
         * Get current min per lane and current index per lane
         * when a min is selected
         */
        p0 = vcmpleq(vecSrc, curExtremValVec);
        curExtremValVec = vpselq(vecSrc, curExtremValVec, p0);
        curExtremIdxVec = vpselq(indexVec, curExtremIdxVec, p0);

        indexVec = indexVec + 8;
        /*
         * Decrement the blockSize loop counter
         */
        blkCnt--;
    }
    /*
     * tail
     * (will be merged thru tail predication)
     */
    blkCnt = blockSize & 7;
    if (blkCnt > 0)
    {
        vecSrc = vldrhq_f16(pSrcVec);  pSrcVec += 8;
        p0 = vctp16q(blkCnt);
        /*
         * Get current min per lane and current index per lane
         * when a min is selected
         */
        p0 = vcmpleq_m(vecSrc, curExtremValVec, p0);
        curExtremValVec = vpselq(vecSrc, curExtremValVec, p0);
        curExtremIdxVec = vpselq(indexVec, curExtremIdxVec, p0);
    }
    /*
     * Get min value across the vector
     */
    minValue = vminnmvq(minValue, curExtremValVec);
    /*
     * set index for lower values to min possible index
     */
    p0 = vcmpleq(curExtremValVec, minValue);
    indexVec = vpselq(curExtremIdxVec, vdupq_n_u16(blockSize), p0);
    /*
     * Get min index which is thus for a min value
     */
    idx = vminvq(idx, indexVec);
    /*
     * Save result
     */
    *pIndex = idx;
    *pResult = minValue;
}

#else

void arm_min_f16(
  const float16_t * pSrc,
        uint32_t blockSize,
        float16_t * pResult,
        uint32_t * pIndex)
{
        float16_t minVal, out;                         /* Temporary variables to store the output value. */
        uint32_t blkCnt, outIndex;                     /* Loop counter */

#if defined (ARM_MATH_LOOPUNROLL) && !defined(ARM_MATH_AUTOVECTORIZE)
        uint32_t index;                                /* index of maximum value */
#endif

  /* Initialise index value to zero. */
  outIndex = 0U;

  /* Load first input value that act as reference value for comparision */
  out = *pSrc++;

#if defined (ARM_MATH_LOOPUNROLL) && !defined(ARM_MATH_AUTOVECTORIZE)
  /* Initialise index of maximum value. */
  index = 0U;

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = (blockSize - 1U) >> 2U;

  while (blkCnt > 0U)
  {
    /* Initialize minVal to next consecutive values one by one */
    minVal = *pSrc++;

    /* compare for the minimum value */
    if ((_Float16)out > (_Float16)minVal)
    {
      /* Update the minimum value and it's index */
      out = minVal;
      outIndex = index + 1U;
    }

    minVal = *pSrc++;
    if ((_Float16)out > (_Float16)minVal)
    {
      out = minVal;
      outIndex = index + 2U;
    }

    minVal = *pSrc++;
    if ((_Float16)out > (_Float16)minVal)
    {
      out = minVal;
      outIndex = index + 3U;
    }

    minVal = *pSrc++;
    if ((_Float16)out > (_Float16)minVal)
    {
      out = minVal;
      outIndex = index + 4U;
    }

    index += 4U;

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = (blockSize - 1U) % 4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = (blockSize - 1U);

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* Initialize minVal to the next consecutive values one by one */
    minVal = *pSrc++;

    /* compare for the minimum value */
    if ((_Float16)out > (_Float16)minVal)
    {
      /* Update the minimum value and it's index */
      out = minVal;
      outIndex = blockSize - blkCnt;
    }

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Store the minimum value and it's index into destination pointers */
  *pResult = out;
  *pIndex = outIndex;
}
#endif /* defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) */

/**
  @} end of Min group
 */

#endif /* #if defined(ARM_FLOAT16_SUPPORTED) */ 

