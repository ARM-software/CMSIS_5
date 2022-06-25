/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_mat_inverse_f64.c
 * Description:  Floating-point matrix inverse
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
#include "dsp/matrix_utils.h"

/**
  @ingroup groupMatrix
 */


/**
  @addtogroup MatrixInv
  @{
 */

/**
  @brief         Floating-point (64 bit) matrix inverse.
  @param[in]     pSrc      points to input matrix structure. The source matrix is modified by the function.
  @param[out]    pDst      points to output matrix structure
  @return        execution status
                   - \ref ARM_MATH_SUCCESS       : Operation successful
                   - \ref ARM_MATH_SIZE_MISMATCH : Matrix size check failed
                   - \ref ARM_MATH_SINGULAR      : Input matrix is found to be singular (non-invertible)
 */

arm_status arm_mat_inverse_f64(
  const arm_matrix_instance_f64 * pSrc,
        arm_matrix_instance_f64 * pDst)
{
  float64_t *pIn = pSrc->pData;                  /* input data matrix pointer */
  float64_t *pOut = pDst->pData;                 /* output data matrix pointer */
  
  float64_t *pTmp;
  uint32_t numRows = pSrc->numRows;              /* Number of rows in the matrix  */
  uint32_t numCols = pSrc->numCols;              /* Number of Cols in the matrix  */


  float64_t pivot = 0.0, newPivot=0.0;                /* Temporary input values  */
  uint32_t selectedRow,pivotRow,i, rowNb, rowCnt, flag = 0U, j,column;      /* loop counters */
  arm_status status;                             /* status of matrix inverse */

#ifdef ARM_MATH_MATRIX_CHECK

  /* Check for matrix mismatch condition */
  if ((pSrc->numRows != pSrc->numCols) ||
      (pDst->numRows != pDst->numCols) ||
      (pSrc->numRows != pDst->numRows)   )
  {
    /* Set status as ARM_MATH_SIZE_MISMATCH */
    status = ARM_MATH_SIZE_MISMATCH;
  }
  else

#endif /* #ifdef ARM_MATH_MATRIX_CHECK */

  {
    /*--------------------------------------------------------------------------------------------------------------
     * Matrix Inverse can be solved using elementary row operations.
     *
     *  Gauss-Jordan Method:
     *
     *      1. First combine the identity matrix and the input matrix separated by a bar to form an
     *        augmented matrix as follows:
     *                      _                  _         _         _
     *                     |  a11  a12 | 1   0  |       |  X11 X12  |
     *                     |           |        |   =   |           |
     *                     |_ a21  a22 | 0   1 _|       |_ X21 X21 _|
     *
     *      2. In our implementation, pDst Matrix is used as identity matrix.
     *
     *      3. Begin with the first row. Let i = 1.
     *
     *      4. Check to see if the pivot for row i is zero.
     *         The pivot is the element of the main diagonal that is on the current row.
     *         For instance, if working with row i, then the pivot element is aii.
     *         If the pivot is zero, exchange that row with a row below it that does not
     *         contain a zero in column i. If this is not possible, then an inverse
     *         to that matrix does not exist.
     *
     *      5. Divide every element of row i by the pivot.
     *
     *      6. For every row below and  row i, replace that row with the sum of that row and
     *         a multiple of row i so that each new element in column i below row i is zero.
     *
     *      7. Move to the next row and column and repeat steps 2 through 5 until you have zeros
     *         for every element below and above the main diagonal.
     *
     *      8. Now an identical matrix is formed to the left of the bar(input matrix, pSrc).
     *         Therefore, the matrix to the right of the bar is our solution(pDst matrix, pDst).
     *----------------------------------------------------------------------------------------------------------------*/

    /* Working pointer for destination matrix */
    pTmp = pOut;

    /* Loop over the number of rows */
    rowCnt = numRows;

    /* Making the destination matrix as identity matrix */
    while (rowCnt > 0U)
    {
      /* Writing all zeroes in lower triangle of the destination matrix */
      j = numRows - rowCnt;
      while (j > 0U)
      {
        *pTmp++ = 0.0;
        j--;
      }

      /* Writing all ones in the diagonal of the destination matrix */
      *pTmp++ = 1.0;

      /* Writing all zeroes in upper triangle of the destination matrix */
      j = rowCnt - 1U;
      while (j > 0U)
      {
        *pTmp++ = 0.0;
        j--;
      }

      /* Decrement loop counter */
      rowCnt--;
    }

    /* Loop over the number of columns of the input matrix.
       All the elements in each column are processed by the row operations */

    /* Index modifier to navigate through the columns */
    for(column = 0U; column < numCols; column++)
    {
      /* Check if the pivot element is zero..
       * If it is zero then interchange the row with non zero row below.
       * If there is no non zero element to replace in the rows below,
       * then the matrix is Singular. */

      pivotRow = column;

      /* Temporary variable to hold the pivot value */
      pTmp = ELEM(pSrc,column,column) ;
      pivot = *pTmp;
      selectedRow = column;

      
        /* Loop over the number rows present below */

      for (rowNb = column+1; rowNb < numRows; rowNb++)
      {
          /* Update the input and destination pointers */
          pTmp = ELEM(pSrc,rowNb,column);
          newPivot = *pTmp;
          if (fabs(newPivot) > fabs(pivot))
          {
            selectedRow = rowNb; 
            pivot = newPivot;
          }
      }

          /* Check if there is a non zero pivot element to
           * replace in the rows below */
      if ((pivot != 0.0) && (selectedRow != column))
      {
            /* Loop over number of columns
             * to the right of the pilot element */

            SWAP_ROWS_F64(pSrc,column, pivotRow,selectedRow);
            SWAP_ROWS_F64(pDst,0, pivotRow,selectedRow);

    
            /* Flag to indicate whether exchange is done or not */
            flag = 1U;

      }


      /* Update the status if the matrix is singular */
      if ((flag != 1U) && (pivot == 0.0))
      {
        return ARM_MATH_SINGULAR;
      }

     
      /* Pivot element of the row */
      pivot = 1.0 / pivot;

      SCALE_ROW_F64(pSrc,column,pivot,pivotRow);
      SCALE_ROW_F64(pDst,0,pivot,pivotRow);

      
      /* Replace the rows with the sum of that row and a multiple of row i
       * so that each new element in column i above row i is zero.*/

      rowNb = 0;
      for (;rowNb < pivotRow; rowNb++)
      {
           pTmp = ELEM(pSrc,rowNb,column) ;
           pivot = *pTmp;

           MAS_ROW_F64(column,pSrc,rowNb,pivot,pSrc,pivotRow);
           MAS_ROW_F64(0     ,pDst,rowNb,pivot,pDst,pivotRow);


      }

      for (rowNb = pivotRow + 1; rowNb < numRows; rowNb++)
      {
           pTmp = ELEM(pSrc,rowNb,column) ;
           pivot = *pTmp;

           MAS_ROW_F64(column,pSrc,rowNb,pivot,pSrc,pivotRow);
           MAS_ROW_F64(0     ,pDst,rowNb,pivot,pDst,pivotRow);

      }

    }

    /* Set status as ARM_MATH_SUCCESS */
    status = ARM_MATH_SUCCESS;

    if ((flag != 1U) && (pivot == 0.0))
    {
      pIn = pSrc->pData;
      for (i = 0; i < numRows * numCols; i++)
      {
        if (pIn[i] != 0.0)
            break;
      }

      if (i == numRows * numCols)
        status = ARM_MATH_SINGULAR;
    }
  }

  /* Return to application */
  return (status);
}
/**
  @} end of MatrixInv group
 */
