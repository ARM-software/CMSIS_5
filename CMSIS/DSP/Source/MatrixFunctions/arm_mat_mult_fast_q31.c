/* ----------------------------------------------------------------------
* Copyright (C) 2010-2014 ARM Limited. All rights reserved.
*
* $Date:        26. October 2016
* $Revision:    V.1.4.5 a
*
* Project:      CMSIS DSP Library
* Title:        arm_mat_mult_fast_q31.c
*
* Description:  Q31 matrix multiplication (fast variant).
*
* Target Processor: Cortex-M4/Cortex-M3
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*   - Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   - Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in
*     the documentation and/or other materials provided with the
*     distribution.
*   - Neither the name of ARM LIMITED nor the names of its contributors
*     may be used to endorse or promote products derived from this
*     software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
* -------------------------------------------------------------------- */

#include "arm_math.h"

/**
 * @ingroup groupMatrix
 */

/**
 * @addtogroup MatrixMult
 * @{
 */

/**
 * @brief Q31 matrix multiplication (fast variant) for Cortex-M3 and Cortex-M4
 * @param[in]       *pSrcA points to the first input matrix structure
 * @param[in]       *pSrcB points to the second input matrix structure
 * @param[out]      *pDst points to output matrix structure
 * @return          The function returns either
 * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
 *
 * @details
 * <b>Scaling and Overflow Behavior:</b>
 *
 * \par
 * The difference between the function arm_mat_mult_q31() and this fast variant is that
 * the fast variant use a 32-bit rather than a 64-bit accumulator.
 * The result of each 1.31 x 1.31 multiplication is truncated to
 * 2.30 format. These intermediate results are accumulated in a 32-bit register in 2.30
 * format. Finally, the accumulator is saturated and converted to a 1.31 result.
 *
 * \par
 * The fast version has the same overflow behavior as the standard version but provides
 * less precision since it discards the low 32 bits of each multiplication result.
 * In order to avoid overflows completely the input signals must be scaled down.
 * Scale down one of the input matrices by log2(numColsA) bits to
 * avoid overflows, as a total of numColsA additions are computed internally for each
 * output element.
 *
 * \par
 * See <code>arm_mat_mult_q31()</code> for a slower implementation of this function
 * which uses 64-bit accumulation to provide higher precision.
 */

arm_status arm_mat_mult_fast_q31(
  const arm_matrix_instance_q31 * pSrcA,
  const arm_matrix_instance_q31 * pSrcB,
  arm_matrix_instance_q31 * pDst)
{
  q31_t *pInA = pSrcA->pData;                    /* input data matrix pointer A */
  q31_t *pInB = pSrcB->pData;                    /* input data matrix pointer B */
  q31_t *px;                                     /* Temporary output data matrix pointer */
  q31_t sum;                                     /* Accumulator */
  uint16_t numRowsA = pSrcA->numRows;            /* number of rows of input matrix A    */
  uint16_t numColsB = pSrcB->numCols;            /* number of columns of input matrix B */
  uint16_t numColsA = pSrcA->numCols;            /* number of columns of input matrix A */
  uint32_t col, i = 0u, j, row = numRowsA, colCnt;  /* loop counters */
  arm_status status;                             /* status of matrix multiplication */
  q31_t inA1, inB1;

#ifndef ARM_MATH_CM0_FAMILY

  q31_t sum2, sum3, sum4;
  q31_t inA2, inB2;
  q31_t *pInA2;
  q31_t *px2;

#endif

#ifdef ARM_MATH_MATRIX_CHECK

  /* Check for matrix mismatch condition */
  if((pSrcA->numCols != pSrcB->numRows) ||
     (pSrcA->numRows != pDst->numRows) || (pSrcB->numCols != pDst->numCols))
  {
    /* Set status as ARM_MATH_SIZE_MISMATCH */
    status = ARM_MATH_SIZE_MISMATCH;
  }
  else
#endif /*      #ifdef ARM_MATH_MATRIX_CHECK    */

  {

    px = pDst->pData;

#ifndef ARM_MATH_CM0_FAMILY
    row = row >> 1;
    px2 = px + numColsB;
#endif

    /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
    /* row loop */
    while(row > 0u)
    {

      /* For every row wise process, the column loop counter is to be initiated */
      col = numColsB;

      /* For every row wise process, the pIn2 pointer is set
       ** to the starting address of the pSrcB data */
      pInB = pSrcB->pData;

      j = 0u;

#ifndef ARM_MATH_CM0_FAMILY
      col = col >> 1;
#endif

      /* column loop */
      while (col > 0u)
      {
        /* Set the variable sum, that acts as accumulator, to zero */
        sum = 0;

        /* Initiate data pointers */
        pInA = pSrcA->pData + i;
        pInB  = pSrcB->pData + j;

#ifndef ARM_MATH_CM0_FAMILY
        sum2 = 0;
        sum3 = 0;
        sum4 = 0;
        pInA2 = pInA + numColsA;
        colCnt = numColsA;
#else
        colCnt = numColsA >> 2;
#endif

        /* matrix multiplication */
        while(colCnt > 0u)
        {

#ifndef ARM_MATH_CM0_FAMILY
          inA1 = *pInA++;
          inB1 = pInB[0];
          inA2 = *pInA2++;
          inB2 = pInB[1];
          pInB += numColsB;

          sum  = __SMMLA(inA1, inB1, sum);
          sum2 = __SMMLA(inA1, inB2, sum2);
          sum3 = __SMMLA(inA2, inB1, sum3);
          sum4 = __SMMLA(inA2, inB2, sum4);
#else
          /* c(m,n) = a(1,1)*b(1,1) + a(1,2) * b(2,1) + .... + a(m,p)*b(p,n) */
          /* Perform the multiply-accumulates */
          inB1 = *pInB;
          pInB += numColsB;
          inA1 = pInA[0];
          sum = __SMMLA(inA1, inB1, sum);

          inB1 = *pInB;
          pInB += numColsB;
          inA1 = pInA[1];
          sum = __SMMLA(inA1, inB1, sum);

          inB1 = *pInB;
          pInB += numColsB;
          inA1 = pInA[2];
          sum = __SMMLA(inA1, inB1, sum);

          inB1 = *pInB;
          pInB += numColsB;
          inA1 = pInA[3];
          sum = __SMMLA(inA1, inB1, sum);

          pInA += 4u;
#endif

          /* Decrement the loop counter */
          colCnt--;
        }

#ifdef ARM_MATH_CM0_FAMILY
        /* If the columns of pSrcA is not a multiple of 4, compute any remaining output samples here. */
        colCnt = numColsA % 0x4u;
        while(colCnt > 0u)
        {
          sum = __SMMLA(*pInA++, *pInB, sum);
          pInB += numColsB;
          colCnt--;
        }
        j++;
#endif

        /* Convert the result from 2.30 to 1.31 format and store in destination buffer */
        *px++  = sum << 1;

#ifndef ARM_MATH_CM0_FAMILY
        *px++  = sum2 << 1;
        *px2++ = sum3 << 1;
        *px2++ = sum4 << 1;
        j += 2;
#endif

        /* Decrement the column loop counter */
        col--;

      }

      i = i + numColsA;

#ifndef ARM_MATH_CM0_FAMILY
      i = i + numColsA;
      px = px2 + (numColsB & 1u);
      px2 = px + numColsB;
#endif

      /* Decrement the row loop counter */
      row--;

    }

    /* Compute any remaining odd row/column below */

#ifndef ARM_MATH_CM0_FAMILY

    /* Compute remaining output column */
    if (numColsB & 1u) {

      /* Avoid redundant computation of last element */
      row = numRowsA & (~0x1);

      /* Point to remaining unfilled column in output matrix */
      px = pDst->pData+numColsB-1;
      pInA = pSrcA->pData;

      /* row loop */
      while (row > 0)
      {

        /* point to last column in matrix B */
        pInB  = pSrcB->pData + numColsB-1;

        /* Set the variable sum, that acts as accumulator, to zero */
        sum  = 0;

        /* Compute 4 columns at once */
        colCnt = numColsA >> 2;

        /* matrix multiplication */
        while(colCnt > 0u)
        {
          inA1 = *pInA++;
          inA2 = *pInA++;
          inB1 = *pInB;
          pInB += numColsB;
          inB2 = *pInB;
          pInB += numColsB;
          sum = __SMMLA(inA1, inB1, sum);
          sum = __SMMLA(inA2, inB2, sum);

          inA1 = *pInA++;
          inA2 = *pInA++;
          inB1 = *pInB;
          pInB += numColsB;
          inB2 = *pInB;
          pInB += numColsB;
          sum = __SMMLA(inA1, inB1, sum);
          sum = __SMMLA(inA2, inB2, sum);

          /* Decrement the loop counter */
          colCnt--;
        }

        colCnt = numColsA & 3u;
        while(colCnt > 0u) {
          sum = __SMMLA(*pInA++, *pInB, sum);
          pInB += numColsB;
          colCnt--;
        }

        /* Convert the result from 2.30 to 1.31 format and store in destination buffer */
        *px = sum << 1;
        px += numColsB;

        /* Decrement the row loop counter */
        row--;
      }
    }

    /* Compute remaining output row */
    if (numRowsA & 1u) {

      /* point to last row in output matrix */
      px = pDst->pData+(numColsB)*(numRowsA-1);

      col = numColsB;
      i = 0u;

      /* col loop */
      while (col > 0)
      {

        /* point to last row in matrix A */
        pInA = pSrcA->pData + (numRowsA-1)*numColsA;
        pInB  = pSrcB->pData + i;

        /* Set the variable sum, that acts as accumulator, to zero */
        sum  = 0;

        /* Compute 4 columns at once */
        colCnt = numColsA >> 2;

        /* matrix multiplication */
        while(colCnt > 0u)
        {
          inA1 = *pInA++;
          inA2 = *pInA++;
          inB1 = *pInB;
          pInB += numColsB;
          inB2 = *pInB;
          pInB += numColsB;
          sum = __SMMLA(inA1, inB1, sum);
          sum = __SMMLA(inA2, inB2, sum);

          inA1 = *pInA++;
          inA2 = *pInA++;
          inB1 = *pInB;
          pInB += numColsB;
          inB2 = *pInB;
          pInB += numColsB;
          sum = __SMMLA(inA1, inB1, sum);
          sum = __SMMLA(inA2, inB2, sum);

          /* Decrement the loop counter */
          colCnt--;
        }

        colCnt = numColsA & 3u;
        while(colCnt > 0u) {
          sum = __SMMLA(*pInA++, *pInB, sum);
          pInB += numColsB;
          colCnt--;
        }

        /* Saturate and store the result in the destination buffer */
        *px++ = sum << 1;
        i++;

        /* Decrement the col loop counter */
        col--;
      }
    }

#endif /* #ifndef ARM_MATH_CM0_FAMILY */

    /* set status as ARM_MATH_SUCCESS */
    status = ARM_MATH_SUCCESS;
  }

  /* Return to application */
  return (status);
}

/**
 * @} end of MatrixMult group
 */
