/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_scale_f64.c
 * Description:  Multiplies a floating-point vector by a scalar
 *
 * $Date:        13 September 2021
 * $Revision:    V1.10.0
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
  @defgroup BasicScale Vector Scale

  Multiply a vector by a scalar value.  For floating-point data, the algorithm used is:

  <pre>
      pDst[n] = pSrc[n] * scale,   0 <= n < blockSize.
  </pre>

  In the fixed-point Q7, Q15, and Q31 functions, <code>scale</code> is represented by
  a fractional multiplication <code>scaleFract</code> and an arithmetic shift <code>shift</code>.
  The shift allows the gain of the scaling operation to exceed 1.0.
  The algorithm used with fixed-point data is:

  <pre>
      pDst[n] = (pSrc[n] * scaleFract) << shift,   0 <= n < blockSize.
  </pre>

  The overall scale factor applied to the fixed-point data is
  <pre>
      scale = scaleFract * 2^shift.
  </pre>

  The functions support in-place computation allowing the source and destination
  pointers to reference the same memory buffer.
 */

/**
  @addtogroup BasicScale
  @{
 */

/**
  @brief         Multiplies a floating-point vector by a scalar.
  @param[in]     pSrc       points to the input vector
  @param[in]     scale      scale factor to be applied
  @param[out]    pDst       points to the output vector
  @param[in]     blockSize  number of samples in each vector
  @return        none
 */

void arm_scale_f64(
  const float64_t *pSrc,
        float64_t scale,
        float64_t *pDst,
        uint32_t blockSize)
{
  uint32_t blkCnt;                               /* Loop counter */

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

  while (blkCnt > 0U)
  {
    /* C = A * scale */

    /* Scale input and store result in destination buffer. */
    *pDst++ = (*pSrc++) * scale;

    /* Decrement loop counter */
    blkCnt--;
  }

}

/**
  @} end of BasicScale group
 */
