/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_iir_2nd_init_f32.c
 * Description:  2nd order IIR filter initialization function
 *
 * $Date:        
 * $Revision:    
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

void arm_iir_2nd_init_f32(
    arm_iir_instance_f32 * S,
    arm_iir_type iirType,
    uint32_t nbCascaded,
    uint16_t simdFlag, 
    float32_t * pState,
    float32_t * pCoeffs )
{
    S->pCoeffs = pCoeffs; // Move [b0 b1 b2 a1 a2] [b0 b1 b2 a1 a2] ...

    switch (iirType)
    {
        case ARM_IIR_DF1:
        /* Clear state buffer */
        memset(pState, 0, nbCascaded*4*sizeof(float32_t));
        S->pState = pState;

#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
#if defined(ARM_MATH_NEON)
        if(simdFlag)
#endif
        {
            float32_t * pDstCoeffs = S->pCoeffs;
        
            /* In order to avoid overwriting of original coefficients, start from last stage */
            pCoeffs += 5*(nbCascaded-1);
            pDstCoeffs += 32*(nbCascaded-1);
        
            while (nbCascaded > 0)
            {
                /* Load coefficients */
                float32_t b0 = pCoeffs[0];
                float32_t b1 = pCoeffs[1];
                float32_t b2 = pCoeffs[2];
                float32_t a1 = pCoeffs[3];
                float32_t a2 = pCoeffs[4];

                /* Update pointer to take b0 b1 b2 a1 a2 of previous stage */
                pCoeffs -= 5;
       
                /* Build system */ 
                float32_t coeffs[4][8] =
                {
                    { 0,  0,  0,  b0, b1, b2, a1, a2 },
                    { 0,  0,  b0, b1, b2, 0,  a2, 0  },
                    { 0,  b0, b1, b2, 0,  0,  0,  0  },
                    { b0, b1, b2, 0,  0,  0,  0,  0  },
                };   
            
                for (int i=0; i<8; i++)
                {
                    /* Add a1*row[0] to row[1] */
                    coeffs[1][i] += a1*coeffs[0][i];
                    /* Add a1*row[1] + a2*row[0] to row 2 */
                    coeffs[2][i] += a1*coeffs[1][i] + a2*coeffs[0][i];
                    /* Add a1*row[2] + a2*row[1] to row 3 */
                    coeffs[3][i] += a1*coeffs[2][i] + a2*coeffs[1][i];
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
            
                /* Coefficients for x[n]   */
                *pDstCoeffs++ = coeffs[0][3];
                *pDstCoeffs++ = coeffs[1][3];
                *pDstCoeffs++ = coeffs[2][3];
                *pDstCoeffs++ = coeffs[3][3];
            
                /* Coefficients for x[n-1] */
                *pDstCoeffs++ = coeffs[0][4];
                *pDstCoeffs++ = coeffs[1][4];
                *pDstCoeffs++ = coeffs[2][4];
                *pDstCoeffs++ = coeffs[3][4];
            
                /* Coefficients for x[n-2] */
                *pDstCoeffs++ = coeffs[0][5];
                *pDstCoeffs++ = coeffs[1][5];
                *pDstCoeffs++ = coeffs[2][5];
                *pDstCoeffs++ = coeffs[3][5];
            
                /* Coefficients for y[n-1] */
                *pDstCoeffs++ = coeffs[0][6];
                *pDstCoeffs++ = coeffs[1][6];
                *pDstCoeffs++ = coeffs[2][6];
                *pDstCoeffs++ = coeffs[3][6];
            
                /* Coefficients for y[n-2] */
                *pDstCoeffs++ = coeffs[0][7];
                *pDstCoeffs++ = coeffs[1][7];
                *pDstCoeffs++ = coeffs[2][7];
                *pDstCoeffs++ = coeffs[3][7];

                /* Update pointers to write for previous stage */
                pDstCoeffs -= 64; 

                nbCascaded--;
            }
        }

#if defined(ARM_MATH_MVEF)
    /* To avoid warning */
    (void)simdFlag;
#endif
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
        /* Clear state buffer */
        memset(pState, 0, nbCascaded*2*sizeof(float32_t));
        S->pState = pState;

#if defined(ARM_MATH_NEON)
        float32_t coeffs_df2T[5*nbCascaded];
        for(int i=0; i<5*(int)nbCascaded; i++)
            coeffs_df2T[i] = pCoeffs[i];

        arm_biquad_cascade_df2T_compute_coefs_f32((arm_biquad_casd_df1_inst_f32 *)S, nbCascaded, coeffs_df2T);
#endif
        break;

    }
}

