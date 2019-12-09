/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_iir_1st_init_f32.c
 * Description:  1st order IIR filter initialization function
 *
 * $Date:        
 * $Revision:    
 *
 * Target Processor: 
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

void arm_iir_1st_init_f32(
    arm_iir_instance_f32 * S,
    arm_iir_type iirType,
    uint32_t nbCascaded,
    uint16_t simdFlag,
    float32_t * pState,
    float32_t * pCoeffs )
{
    S->pCoeffs = pCoeffs; // Move [b0 b1 a1] [b0 b1 a1] ...

    switch (iirType)
    {
        case ARM_IIR_DF1:
        /* Clear state buffer */
        memset(pState, 0, nbCascaded*2*sizeof(float32_t));
        S->pState = pState;
#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
        if(simdFlag)
        {
            float32_t * pDstCoeffs = S->pCoeffs;
            /* In order to avoid overwriting of original coefficients, start from last stage */
            pCoeffs += 3*(nbCascaded-1);
            pDstCoeffs += 24*(nbCascaded-1);
    
            while (nbCascaded>0)
            {
                /* Load coefficients */
                float32_t b0 = pCoeffs[0];
                float32_t b1 = pCoeffs[1];
                float32_t a1 = pCoeffs[2];
    
                /* Update pointer to b0 b1 a1 of previous stage */
                pCoeffs -= 3;
        
                /* Build system */
                float32_t coeffs[4][6] =
                {
                    { 0,  0,  0,  b0, b1, a1 },
                    { 0,  0,  b0, b1, 0,  0  },
                    { 0,  b0, b1, 0,  0,  0  },
                    { b0, b1, 0,  0,  0,  0  },
                };   
            
                for (int i=0; i<6; i++)
                {
                    /* Add a1*row[0] to row 1 */
                    coeffs[1][i] += a1 * coeffs[0][i];
                    /* Add a1*row[1] to row 2 */
                    coeffs[2][i] += a1 * coeffs[1][i];
                    /* Add a1*row[2] to row 3 */
                    coeffs[3][i] += a1 * coeffs[2][i];
                }
            
                /* Coefficients for x[n+3] */
                *pDstCoeffs++ = coeffs[0][0];
                *pDstCoeffs++ = coeffs[1][0];
                *pDstCoeffs++ = coeffs[2][0];
                *pDstCoeffs++ = coeffs[3][0];
            
                /* Coefficients for x[n+2] */
                *pDstCoeffs++ = coeffs[0][1];
                *pDstCoeffs++ = coeffs[1][1];
                *pDstCoeffs++ = coeffs[2][1];
                *pDstCoeffs++ = coeffs[3][1];
            
                /* Coefficients for x[n+1] */
                *pDstCoeffs++ = coeffs[0][2];
                *pDstCoeffs++ = coeffs[1][2];
                *pDstCoeffs++ = coeffs[2][2];
                *pDstCoeffs++ = coeffs[3][2];
            
                /* Coefficients for x[n] */
                *pDstCoeffs++ = coeffs[0][3];
                *pDstCoeffs++ = coeffs[1][3];
                *pDstCoeffs++ = coeffs[2][3];
                *pDstCoeffs++ = coeffs[3][3];
            
                /* Coefficients for x[n-1] */
                *pDstCoeffs++ = coeffs[0][4];
                *pDstCoeffs++ = coeffs[1][4];
                *pDstCoeffs++ = coeffs[2][4];
                *pDstCoeffs++ = coeffs[3][4];
            
                /* Coefficients for y[n-1] */
                *pDstCoeffs++ = coeffs[0][5];
                *pDstCoeffs++ = coeffs[1][5];
                *pDstCoeffs++ = coeffs[2][5];
                *pDstCoeffs++ = coeffs[3][5];
       
                /* Update pointer to write coefficients of previous stage */
                pDstCoeffs -= 48;

                nbCascaded--;
            }    
        }
#else
        /* To avoid warning */
        (void)simdFlag;
#endif
        break;

        case ARM_IIR_DF2:
        /* Not implemented */
        break;

        case ARM_IIR_DF1T:
        /* Not implemented */
        break;

        case ARM_IIR_DF2T:
        /* Not implemented */
        break;
    }

}
