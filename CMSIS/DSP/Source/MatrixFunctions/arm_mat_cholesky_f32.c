/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_mat_cholesky_f32.c
 * Description:  Floating-point matrix Cholesky decomposition
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
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
#include "arm_helium_utils.h"
#endif
#if defined(ARM_ERROR_HANDLER)
#include "arm_error.h"
#endif

/**
  @ingroup groupMatrix
 */

/**
  @defgroup CholeskyDec Cholesky Decomposition
  
  Compute the Cholesky decomposition of a matrix.

  This set of functions compute the Cholesky decomposition of a Hermitian positive
  definite matrix, useful for numerical solutions.
  
 */

/**
  @addtogroup CholeskyDec
  @{
 */

/**
 * @brief         Floating-point Cholesky decomposition.
 * @param[in]     pSrc      points to input matrix structure
 * @param[out]    pDst      points to output matrix structure
 * @return        none
 *
 * @par          Description
 *                 The function takes as input the pointer to 
 *                 the Hermitian positive definite matrix
 *                 A and returns the pointer to a lower 
 *                 triangular matrix L, where A = L L* (L* is 
 *                 the conjugate transpose of L). 
 *
 */

void arm_mat_cholesky_f32(
  const arm_matrix_instance_f32 * pSrc,
        arm_matrix_instance_f32 * pDst)
{
    float32_t *pIn  = pSrc->pData;      /* Input data matrix pointer  */
    float32_t *pOut = pDst->pData;      /* Output data matrix pointer */

#if defined (ARM_ERROR_HANDLER)
    if( (pSrc->numRows) != (pSrc->numCols) )
       arm_error_handler(ARM_ERROR_MATH, "Source matrix is not square."); 
#endif

    uint32_t n=(uint32_t)pSrc->numRows; 
    uint32_t i, j;
    uint32_t kCnt;
    const float32_t * pVec1;
    const float32_t * pVec2;
#if defined(ARM_MATH_NEON)
    float32x4_t vec1, vec2, accum;
    float32x2_t sum;
#endif
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    f32x4_t vec1, vec2, accum;
    float32_t sum;
#endif

    for(i=0; i<n; i++)
    {
        for(j=i; j<n; j++)
        {
            /* Initialize value */
            pOut[j*n+i] = pIn[j*n+i];

            pVec1 = pOut+(i*n);
            pVec2 = pOut+(j*n);

#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
            /* Compute 4 outputs at a time */
            kCnt = i >> 2;

            /* Intialize accumulator */
            accum = vdupq_n_f32(0);

            while(kCnt>0)
            {
                /* Load input vectors */
                vec1 = vld1q_f32(pVec1);
                vec2 = vld1q_f32(pVec2);

#if defined(ARM_MATH_NEON)
                accum = vmlaq_f32(accum, vec1, vec2);
#else 
                accum = vfmaq_f32(accum, vec1, vec2);
#endif

                pVec1 += 4;
                pVec2 += 4;

                kCnt--;
            }

#if defined(ARM_MATH_NEON)
            sum = vpadd_f32(vget_low_f32(accum), vget_high_f32(accum));
            pOut[j*n+i] -= (sum[0] + sum[1]);
#else
            sum = vecAddAcrossF32Mve(accum);
            pOut[j*n+i] -= sum;
#endif
            /* Tail */
            kCnt = i & 3;
#else
            kCnt = i;            
#endif
            while(kCnt>0)
            {
                pOut[j*n+i] -= (*pVec1)*(*pVec2);

                pVec1++;
                pVec2++;
                
                kCnt--;
            }

            if(i==j)
                arm_sqrt_f32(pOut[i*n+i], pOut+(i*n+i)); /* Diagonal elements */
            else
                pOut[j*n+i] /= pOut[i*n+i];

        }
    }
}

/**
  @} end of CholeskyDec group
 */
