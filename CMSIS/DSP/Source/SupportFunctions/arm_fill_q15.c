/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_fill_q15.c
 * Description:  Fills a constant value into a Q15 vector
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

#include "dsp/support_functions.h"

/**
  @ingroup groupSupport
 */

/**
  @addtogroup Fill
  @{
 */

/**
  @brief         Fills a constant value into a Q15 vector.
  @param[in]     value      input value to be filled
  @param[out]    pDst       points to output vector
  @param[in]     blockSize  number of samples in each vector
  @return        none
 */
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
void arm_fill_q15(
  q15_t value,
  q15_t * pDst,
  uint32_t blockSize)
{
  uint32_t blkCnt;  
  blkCnt = blockSize >> 3;
  while (blkCnt > 0U)
  {

        vstrhq_s16(pDst,vdupq_n_s16(value));
        /*
         * Decrement the blockSize loop counter
         * Advance vector source and destination pointers
         */
        pDst += 8;
        blkCnt --;
    }

  blkCnt = blockSize & 7;
  while (blkCnt > 0U)
  {
    /* C = value */

    /* Fill value in destination buffer */
    *pDst++ = value;

    /* Decrement loop counter */
    blkCnt--;
  }
}

#else
void arm_fill_q15(
  q15_t value,
  q15_t * pDst,
  uint32_t blockSize)
{
  uint32_t blkCnt;                               /* Loop counter */

#if defined (ARM_MATH_LOOPUNROLL)
  q31_t packedValue;                             /* value packed to 32 bits */

  /* Packing two 16 bit values to 32 bit value in order to use SIMD */
  packedValue = __PKHBT(value, value, 16U);

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = value */

    /* fill 2 times 2 samples at a time */
    write_q15x2_ia (&pDst, packedValue);
    write_q15x2_ia (&pDst, packedValue);

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
    /* C = value */

    /* Fill value in destination buffer */
    *pDst++ = value;

    /* Decrement loop counter */
    blkCnt--;
  }
}
#endif /* defined(ARM_MATH_MVEI) */

/**
  @} end of Fill group
 */
