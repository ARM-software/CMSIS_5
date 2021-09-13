/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_power_f64.c
 * Description:  Sum of the squares of the elements of a floating-point vector
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

#include "dsp/statistics_functions.h"

/**
  @ingroup groupStats
 */

/**
  @defgroup power Power

  Calculates the sum of the squares of the elements in the input vector.
  The underlying algorithm is used:

  <pre>
      Result = pSrc[0] * pSrc[0] + pSrc[1] * pSrc[1] + pSrc[2] * pSrc[2] + ... + pSrc[blockSize-1] * pSrc[blockSize-1];
  </pre>

  There are separate functions for floating point, Q31, Q15, and Q7 data types.

  Since the result is not divided by the length, those functions are in fact computing
  something which is more an energy than a power.

 */

/**
  @addtogroup power
  @{
 */

/**
  @brief         Sum of the squares of the elements of a floating-point vector.
  @param[in]     pSrc       points to the input vector
  @param[in]     blockSize  number of samples in input vector
  @param[out]    pResult    sum of the squares value returned here
  @return        none
 */
void arm_power_f64(
  const float64_t * pSrc,
        uint32_t blockSize,
        float64_t * pResult)
{
        uint32_t blkCnt;                               /* Loop counter */
        float64_t sum = 0.0f;                          /* Temporary result storage */
        float64_t in;                                  /* Temporary variable to store input value */

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

  while (blkCnt > 0U)
  {
    /* C = A[0] * A[0] + A[1] * A[1] + ... + A[blockSize-1] * A[blockSize-1] */

    /* Compute Power and store result in a temporary variable, sum. */
    in = *pSrc++;
    sum += in * in;

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Store result to destination */
  *pResult = sum;
}

/**
  @} end of power group
 */
