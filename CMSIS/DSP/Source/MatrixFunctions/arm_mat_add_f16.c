/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_mat_add_f16.c
 * Description:  Floating-point matrix addition
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

#include "dsp/matrix_functions_f16.h"

#if defined(ARM_FLOAT16_SUPPORTED)


/**
  @ingroup groupMatrix
 */


/**
  @addtogroup MatrixAdd
  @{
 */


/**
  @brief         Floating-point matrix addition.
  @param[in]     pSrcA      points to first input matrix structure
  @param[in]     pSrcB      points to second input matrix structure
  @param[out]    pDst       points to output matrix structure
  @return        execution status
                   - \ref ARM_MATH_SUCCESS       : Operation successful
                   - \ref ARM_MATH_SIZE_MISMATCH : Matrix size check failed
 */

#if defined(ARM_MATH_MVE_FLOAT16) && !defined(ARM_MATH_AUTOVECTORIZE)

arm_status arm_mat_add_f16(
  const arm_matrix_instance_f16 * pSrcA,
  const arm_matrix_instance_f16 * pSrcB,
  arm_matrix_instance_f16 * pDst)
{
    arm_status status;  
    uint32_t  numSamples;       /* total number of elements in the matrix  */
    float16_t *pDataA, *pDataB, *pDataDst;
    f16x8_t vecA, vecB, vecDst;
    float16_t const *pSrcAVec;
    float16_t const *pSrcBVec;
    uint32_t  blkCnt;           /* loop counters */

    pDataA = pSrcA->pData;
    pDataB = pSrcB->pData;
    pDataDst = pDst->pData;
    pSrcAVec = (float16_t const *) pDataA;
    pSrcBVec = (float16_t const *) pDataB;

#ifdef ARM_MATH_MATRIX_CHECK
  /* Check for matrix mismatch condition */
  if ((pSrcA->numRows != pSrcB->numRows) ||
     (pSrcA->numCols != pSrcB->numCols) ||
     (pSrcA->numRows != pDst->numRows) || (pSrcA->numCols != pDst->numCols))
  {
    /* Set status as ARM_MATH_SIZE_MISMATCH */
    status = ARM_MATH_SIZE_MISMATCH;
  }
  else
#endif
 {
    /*
     * Total number of samples in the input matrix
     */
    numSamples = (uint32_t) pSrcA->numRows * pSrcA->numCols;
    blkCnt = numSamples >> 3;
    while (blkCnt > 0U)
    {
        /* C(m,n) = A(m,n) + B(m,n) */
        /* Add and then store the results in the destination buffer. */
        vecA = vld1q(pSrcAVec); 
        pSrcAVec += 8;
        vecB = vld1q(pSrcBVec); 
        pSrcBVec += 8;
        vecDst = vaddq(vecA, vecB);
        vst1q(pDataDst, vecDst);  
        pDataDst += 8;
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
        vecA = vld1q(pSrcAVec); 
        vecB = vld1q(pSrcBVec); 
        vecDst = vaddq_m(vecDst, vecA, vecB, p0);
        vstrhq_p(pDataDst, vecDst, p0);
    }
    /* set status as ARM_MATH_SUCCESS */
    status = ARM_MATH_SUCCESS;
  }
  return (status);
}
#else

arm_status arm_mat_add_f16(
  const arm_matrix_instance_f16 * pSrcA,
  const arm_matrix_instance_f16 * pSrcB,
        arm_matrix_instance_f16 * pDst)
{
  float16_t *pInA = pSrcA->pData;                /* input data matrix pointer A */
  float16_t *pInB = pSrcB->pData;                /* input data matrix pointer B */
  float16_t *pOut = pDst->pData;                 /* output data matrix pointer */

  uint32_t numSamples;                           /* total number of elements in the matrix */
  uint32_t blkCnt;                               /* loop counters */
  arm_status status;                             /* status of matrix addition */

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
      /* C(m,n) = A(m,n) + B(m,n) */

      /* Add and store result in destination buffer. */
      *pOut++ = *pInA++ + *pInB++;

      *pOut++ = *pInA++ + *pInB++;

      *pOut++ = *pInA++ + *pInB++;

      *pOut++ = *pInA++ + *pInB++;

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
      /* C(m,n) = A(m,n) + B(m,n) */

      /* Add and store result in destination buffer. */
      *pOut++ = *pInA++ + *pInB++;

      /* Decrement loop counter */
      blkCnt--;
    }

    /* Set status as ARM_MATH_SUCCESS */
    status = ARM_MATH_SUCCESS;
  }

  /* Return to application */
  return (status);
}
#endif /* defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) */

/**
  @} end of MatrixAdd group
 */

#endif /* #if defined(ARM_FLOAT16_SUPPORTED) */ 

