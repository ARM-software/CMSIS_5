/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_iir_generic_init_f32.c
 * Description:  Generic IIR filter initialization function
 *
 * $Date:        2019
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

void arm_iir_generic_init_f32(
    arm_iir_instance_f32 * S,
    arm_iir_type iirType,
    uint32_t nbCascaded,
    uint16_t simdFlag,
    uint16_t order,
    float32_t * pState,
    float32_t * pCoeffs )
{
    S->pCoeffs = pCoeffs;

    switch (iirType)
    {
        case ARM_IIR_DF1:
        /* Clear state buffer */
        memset(pState, 0, 2*nbCascaded*order*sizeof(float32_t));
        S->pState = pState;

#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
        uint16_t numCoeff = 2*order + 1;
        uint16_t numCoeffDst = 16+8*order;
        int ii, jj; 
     
        if(simdFlag)
        {
            float32_t * pDstCoeffs = S->pCoeffs;
    
            // In order to avoid overwriting of original coefficients, start from last stage
            pCoeffs += numCoeff*(nbCascaded-1);
            pDstCoeffs += numCoeffDst*(nbCascaded-1);
    
            while (nbCascaded>0)
            {
                /* b0 b1 ... a1 ... */
                float32_t coeff_tmp[numCoeff];
                memcpy(coeff_tmp, pCoeffs, numCoeff*sizeof(float32_t));
    
                /* Cleare area for new coefficients */
                memset(pDstCoeffs, 0, numCoeffDst*sizeof(float32_t) );

                /* Update pointer to coefficients of previous stage */
                pCoeffs -= numCoeff;
                
                /* Create system: */
                /*
                | 0  0  0  b0
                | 0  0  b0 b1
                | 0  b0 b1 b2
                | b0 b1 b2 b3 */
                for (ii = 1; ii <= 4; ii++)
                    memcpy(pDstCoeffs+(3*ii), coeff_tmp, (ii)*sizeof(float32_t));

                /*
                  b1 b2 ... bord-3
                  b2 b3 ... bord-2
                  b3 b4 ... bord-1
                  b4 b5 ... border
                 */
                for (ii = 0; ii <order-3; ii++) // 1 for order 4, 2 for order 5 ...
                    memcpy(pDstCoeffs+(16+ii*4), coeff_tmp+ii+1, 4*sizeof(float32_t));

                /*
                 bord-2 bord-1 border 
                 bord-1 border   0   
                 border   0      0   
                   0      0      0   
                 */
                memcpy(pDstCoeffs+4*(order+1)+0, coeff_tmp+(order-2+0), 3*sizeof(float32_t));
                memcpy(pDstCoeffs+4*(order+1)+4, coeff_tmp+(order-2+1), 2*sizeof(float32_t));
                memcpy(pDstCoeffs+4*(order+1)+8, coeff_tmp+(order-2+2), 1*sizeof(float32_t));
    
                /* 
                 a1 ... aord-3 aord-2 aord-1 aorder |
                 a2 ... aord-2 aord-1 aorder   0    |
                 a3 ... aord-1 aorder   0      0    |
                 a4 ... aorder   0      0      0    |
                 */
                ii=order;
                jj = 0;
                while(ii>0)
                {
                    memcpy(pDstCoeffs+(16+4*(order-3)+12)+jj*4, coeff_tmp+order+1+jj, ii*sizeof(float32_t));
                    ii--;
                    jj++;
                }
    
                for ( ii = 0; ii < numCoeffDst; ii+=4)
                {
                    /* Add a1*row[0] to row 1 */
                    pDstCoeffs[ii+1] += coeff_tmp[order+1]*pDstCoeffs[ii];
            
                    /* Add a1*row[1] + a2*row[0] to row 2 */
                    pDstCoeffs[ii+2] += coeff_tmp[order+1]*pDstCoeffs[ii+1] + coeff_tmp[order+2]*pDstCoeffs[ii];
            
                    /* Add a1*row[2] + a2*row[1] + a3*row[3] to row 3 */
                    pDstCoeffs[ii+3] += coeff_tmp[order+1]*pDstCoeffs[ii+2] + coeff_tmp[order+2]*pDstCoeffs[ii+1] + coeff_tmp[order+3]*pDstCoeffs[ii];
                }
    
                /* Update pointer to write coefficients of previous stage */
                pDstCoeffs -= numCoeffDst;

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

