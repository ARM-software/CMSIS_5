/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_q7_to_float.c
 * Description:  Converts the elements of the Q7 vector to floating-point vector
 *
 * $Date:        28. February 2019
 * $Revision:    V.1.5.5
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
  @ingroup groupSupport
 */

/**
 * @defgroup q7_to_x  Convert 8-bit Integer value
 */

/**
  @addtogroup q7_to_x
  @{
 */

/**
  @brief         Converts the elements of the Q7 vector to floating-point vector.
  @param[in]     pSrc       points to the Q7 input vector
  @param[out]    pDst       points to the floating-point output vector
  @param[in]     blockSize  number of samples in each vector
  @return        none

 @par            Details
                   The equation used for the conversion process is:
  <pre>
      pDst[n] = (float32_t) pSrc[n] / 128;   0 <= n < blockSize.
  </pre>
 */

void arm_q7_to_float(
  const q7_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize)
{
        uint32_t blkCnt;                               /* Loop counter */
  const q7_t *pIn = pSrc;                              /* Source pointer */

#if defined (ARM_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = (float32_t) A / 128 */

    /* Convert from q7 to float and store result in destination buffer */
    *pDst++ = ((float32_t) * pIn++ / 128.0f);
    *pDst++ = ((float32_t) * pIn++ / 128.0f);
    *pDst++ = ((float32_t) * pIn++ / 128.0f);
    *pDst++ = ((float32_t) * pIn++ / 128.0f);

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
    /* C = (float32_t) A / 128 */

    /* Convert from q7 to float and store result in destination buffer */
    *pDst++ = ((float32_t) * pIn++ / 128.0f);

    /* Decrement loop counter */
    blkCnt--;
  }

}

/**
  @} end of q7_to_x group
 */
