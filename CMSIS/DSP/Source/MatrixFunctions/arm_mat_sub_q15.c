/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_mat_sub_q15.c
 * Description:  Q15 Matrix subtraction
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

#include "dsp/matrix_functions.h"

/**
  @ingroup groupMatrix
 */

/**
  @addtogroup MatrixSub
  @{
 */

/**
  @brief         Q15 matrix subtraction.
  @param[in]     pSrcA      points to the first input matrix structure
  @param[in]     pSrcB      points to the second input matrix structure
  @param[out]    pDst       points to output matrix structure
  @return        execution status
                   - \ref ARM_MATH_SUCCESS       : Operation successful
                   - \ref ARM_MATH_SIZE_MISMATCH : Matrix size check failed

  @par           Scaling and Overflow Behavior
                   The function uses saturating arithmetic.
                   Results outside of the allowable Q15 range [0x8000 0x7FFF] are saturated.
 */
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)

arm_status arm_mat_sub_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst)
{
    uint32_t        numSamples;       /* total number of elements in the matrix  */
    q15_t          *pDataA, *pDataB, *pDataDst;
    q15x8_t       vecA, vecB, vecDst;
    q15_t const   *pSrcAVec;
    q15_t const   *pSrcBVec;
    uint32_t        blkCnt;           /* loop counters */
    arm_status status;                             /* status of matrix subtraction  */


    pDataA = pSrcA->pData;
    pDataB = pSrcB->pData;
    pDataDst = pDst->pData;
    pSrcAVec = (q15_t const *) pDataA;
    pSrcBVec = (q15_t const *) pDataB;

  #ifdef ARM_MATH_MATRIX_CHECK

  /* Check for matrix mismatch condition */
  if ((pSrcA->numRows != pSrcB->numRows) ||
      (pSrcA->numCols != pSrcB->numCols) ||
      (pSrcA->numRows != pDst->numRows)  ||
      (pSrcA->numCols != pDst->numCols)    )
  {
    /* Set status as ARM_MATH_SIZE_MISMATCH */
    status = ARM_MATH_SIZE_MISMATCH;
  }
  else
#endif /* #ifdef ARM_MATH_MATRIX_CHECK */

  {
        /*
     * Total number of samples in the input matrix
     */
    numSamples = (uint32_t) pSrcA->numRows * pSrcA->numCols;
    blkCnt = numSamples >> 3;
    while (blkCnt > 0U)
    {
        /* C(m,n) = A(m,n) + B(m,n) */
        /* sub and then store the results in the destination buffer. */
        vecA = vld1q(pSrcAVec); pSrcAVec += 8;
        vecB = vld1q(pSrcBVec); pSrcBVec += 8;
        vecDst = vqsubq(vecA, vecB);
        vst1q(pDataDst, vecDst);  pDataDst += 8;
        /*
         * Decrement the blockSize loop counter
         */
        blkCnt--;
    }
    /*
     * tail
     */
    blkCnt = numSamples & 7;
    if (blkCnt > 0U)
    {
        mve_pred16_t p0 = vctp16q(blkCnt);
        vecA = vld1q(pSrcAVec); pSrcAVec += 8;
        vecB = vld1q(pSrcBVec); pSrcBVec += 8;
        vecDst = vqsubq_m(vecDst, vecA, vecB, p0);
        vstrhq_p(pDataDst, vecDst, p0);
    }
     status = ARM_MATH_SUCCESS;
  }

  /* Return to application */
  return (status);
}

#else
arm_status arm_mat_sub_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst)
{
        q15_t *pInA = pSrcA->pData;                    /* input data matrix pointer A */
        q15_t *pInB = pSrcB->pData;                    /* input data matrix pointer B */
        q15_t *pOut = pDst->pData;                     /* output data matrix pointer */

        uint32_t numSamples;                           /* total number of elements in the matrix */
        uint32_t blkCnt;                               /* loop counters  */
        arm_status status;                             /* status of matrix subtraction  */

#ifdef ARM_MATH_MATRIX_CHECK

  /* Check for matrix mismatch condition */
  if ((pSrcA->numRows != pSrcB->numRows) ||
      (pSrcA->numCols != pSrcB->numCols) ||
      (pSrcA->numRows != pDst->numRows)  ||
      (pSrcA->numCols != pDst->numCols)    )
  {
    /* Set status as ARM_MATH_SIZE_MISMATCH */
    status = ARM_MATH_SIZE_MISMATCH;
  }
  else
#endif /* #ifdef ARM_MATH_MATRIX_CHECK */

  {
    /* Total number of samples in input matrix */
    numSamples = (uint32_t) pSrcA->numRows * pSrcA->numCols;

#if defined (ARM_MATH_LOOPUNROLL)

    /* Loop unrolling: Compute 4 outputs at a time */
    blkCnt = numSamples >> 2U;

    while (blkCnt > 0U)
    {
      /* C(m,n) = A(m,n) - B(m,n) */

      /* Subtract, Saturate and store result in destination buffer. */
#if defined (ARM_MATH_DSP)
      write_q15x2_ia (&pOut, __QSUB16(read_q15x2_ia (&pInA), read_q15x2_ia (&pInB)));
      write_q15x2_ia (&pOut, __QSUB16(read_q15x2_ia (&pInA), read_q15x2_ia (&pInB)));
#else
      *pOut++ = (q15_t) __SSAT(((q31_t) * pInA++ - *pInB++), 16);
      *pOut++ = (q15_t) __SSAT(((q31_t) * pInA++ - *pInB++), 16);
      *pOut++ = (q15_t) __SSAT(((q31_t) * pInA++ - *pInB++), 16);
      *pOut++ = (q15_t) __SSAT(((q31_t) * pInA++ - *pInB++), 16);
#endif

      /* Decrement loop counter */
      blkCnt--;
    }

    /* Loop unrolling: Compute remaining outputs */
    blkCnt = numSamples % 0x4U;

#else

    /* Initialize blkCnt with number of samples */
    blkCnt = numSamples;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

    while (blkCnt > 0U)
    {
      /* C(m,n) = A(m,n) - B(m,n) */

      /* Subtract and store result in destination buffer. */
#if defined (ARM_MATH_DSP)
      *pOut++ = (q15_t) __QSUB16(*pInA++, *pInB++);
#else
      *pOut++ = (q15_t) __SSAT(((q31_t) * pInA++ - *pInB++), 16);
#endif

      /* Decrement loop counter */
      blkCnt--;
    }

    /* Set status as ARM_MATH_SUCCESS */
    status = ARM_MATH_SUCCESS;
  }

  /* Return to application */
  return (status);
}
#endif /* defined(ARM_MATH_MVEI) */

/**
  @} end of MatrixSub group
 */
