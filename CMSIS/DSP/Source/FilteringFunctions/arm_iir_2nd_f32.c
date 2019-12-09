/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_iir_2nd_f32.c
 * Description:  2nd order IIR filter processing function
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
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
#include "arm_helium_utils.h"
#endif

#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
static void arm_iir_2nd_df1_simd_f32(
    const arm_iir_instance_f32 * S,
          float32_t * pSrc,
          float32_t * pDst,
          uint32_t blockSize)
{
    uint32_t sample;
    float32_t x0, xm1, xm2, y0, ym1, ym2;
    float32_t b0, b1, b2, a1, a2;
    uint32_t nbCascaded = S->numStages;
    float32_t * pState = S->pState;
    float32_t * pCoeffs = S->pCoeffs;
    float32_t * pIn  = pSrc;
    float32_t * pOut;

    /* In-place */
    if(pSrc == pDst)
        pOut = pSrc;
    else
        pOut = pDst;

    /* x[n] x[n+1] x[n+2] x[n+3] */
    float32x4_t x0123;
    /* y[n] y[n+1] y[n+2] y[n+3] */
    float32x4_t y0123;

    /* x[n] x[n] x[n] x[n] */
    float32x4_t value_x0;
    /* x[n+1] x[n+1] x[n+1] x[n+1] */
    float32x4_t value_xp1;
    /* x[n+2] x[n+2] x[n+2] x[n+2] */
    float32x4_t value_xp2;
    /* x[n+3] x[n+3] x[n+3] x[n+3] */
    float32x4_t value_xp3;

    /* x[n-2] x[n-2] x[n-2] x[n-2] */
    float32x4_t value_xm2;
    /* x[n-1] x[n-1] x[n-1] x[n-1] */
    float32x4_t value_xm1;
    /* y[n-2] y[n-2] y[n-2] y[n-2] */
    float32x4_t value_ym2;
    /* y[n-1] y[n-1] y[n-1] y[n-1] */
    float32x4_t value_ym1;

    /* Coefficients */
    float32x4_t coeff_xp3;
    float32x4_t coeff_xp2;
    float32x4_t coeff_xp1;
    float32x4_t coeff_x0 ;
    float32x4_t coeff_xm1;
    float32x4_t coeff_xm2;
    float32x4_t coeff_ym1;
    float32x4_t coeff_ym2;

    while(nbCascaded > 0)
    {
        /* Load coefficients */
        b0 = pCoeffs[12];
        b1 = pCoeffs[16];
        b2 = pCoeffs[20];
        a1 = pCoeffs[24];
        a2 = pCoeffs[28];
    
        /* Load x[n-1], x[n-2], y[n-1], y[n-2] */
        xm1 = pState[0];
        xm2 = pState[1];
        ym1 = pState[2];
        ym2 = pState[3];

        coeff_xp3 = vld1q_f32(pCoeffs+0 );
        coeff_xp2 = vld1q_f32(pCoeffs+4 );
        coeff_xp1 = vld1q_f32(pCoeffs+8 );
        coeff_x0  = vld1q_f32(pCoeffs+12);
        coeff_xm1 = vld1q_f32(pCoeffs+16);
        coeff_xm2 = vld1q_f32(pCoeffs+20);
        coeff_ym1 = vld1q_f32(pCoeffs+24);
        coeff_ym2 = vld1q_f32(pCoeffs+28);
 
        value_xm2 = vdupq_n_f32(xm2);
        value_xm1 = vdupq_n_f32(xm1);
        value_ym2 = vdupq_n_f32(ym2);
        value_ym1 = vdupq_n_f32(ym1);
    
        /* Compute 4 outputs at a time */
        sample = blockSize >> 2U;
    
        while (sample > 0U)
        {
            /* Load x[n] x[n+1] x[n+2] x[n+3] */
            x0123 = vld1q_f32(pIn);
            pIn += 4;
   
            /* Create vectors for x[n], x[n+1], x[n+2], x[n+3] */ 
            value_x0  = vdupq_n_f32(x0123[0]);
            value_xp1 = vdupq_n_f32(x0123[1]);
            value_xp2 = vdupq_n_f32(x0123[2]);
            value_xp3 = vdupq_n_f32(x0123[3]);

            y0123 = vmulq_f32(coeff_xp3, value_xp3);        
#if defined(ARM_MATH_NEON) 
            y0123 = vmlaq_f32(y0123, coeff_xp2, value_xp2);
            y0123 = vmlaq_f32(y0123, coeff_xp1, value_xp1);
            y0123 = vmlaq_f32(y0123, coeff_x0 , value_x0 );
            y0123 = vmlaq_f32(y0123, coeff_xm1, value_xm1);
            y0123 = vmlaq_f32(y0123, coeff_xm2, value_xm2);
            y0123 = vmlaq_f32(y0123, coeff_ym1, value_ym1);
            y0123 = vmlaq_f32(y0123, coeff_ym2, value_ym2);
#endif
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            y0123 = vfmaq_f32(y0123, coeff_xp2, value_xp2);
            y0123 = vfmaq_f32(y0123, coeff_xp1, value_xp1);
            y0123 = vfmaq_f32(y0123, coeff_x0 , value_x0 );
            y0123 = vfmaq_f32(y0123, coeff_xm1, value_xm1);
            y0123 = vfmaq_f32(y0123, coeff_xm2, value_xm2);
            y0123 = vfmaq_f32(y0123, coeff_ym1, value_ym1);
            y0123 = vfmaq_f32(y0123, coeff_ym2, value_ym2);
#endif
            /* Store y[n] y[n+1] y[n+2] y[n+3] */
            vst1q_f32(pOut, y0123);
            pOut += 4;
    
            /* Update state */
            value_xm2 = value_xp2; /* x[n+2] -> x[n-2] */
            value_xm1 = value_xp3; /* x[n+3] -> x[n-1] */
            value_ym2 = vdupq_n_f32(y0123[2]); /* y[n+2] -> y[n-2] */
            value_ym1 = vdupq_n_f32(y0123[3]); /* y[n+3] -> y[n-1] */
    
            sample--;
        }
    
        /* Tail */
        sample = blockSize & 0x3U;
    
        xm2 = value_xp2[0];
        xm1 = value_xp3[0];
        ym2 = y0123[2];
        ym1 = y0123[3];

        while (sample > 0U)
        {
            x0 = *pIn++;
            
            y0 = b0*x0 + b1*xm1 + b2*xm2 + a1*ym1 + a2*ym2;
            
            *pOut++ = y0;
            
            xm2 = xm1;
            xm1 = x0;
            ym2 = ym1;
            ym1 = y0;
    
            sample--;
        }
    
        /* Store the updated state variables back into the pState array */
        pState[0] = xm1;
        pState[1] = xm2;
        pState[2] = ym1;
        pState[3] = ym2;

        pState +=4;

        pOut-=blockSize;
        pIn = pOut;

        pCoeffs += 32;   

        nbCascaded--;
    }
}
#endif

void arm_iir_2nd_f32(
    const arm_iir_instance_f32 * S,
          float32_t * pSrc,
          float32_t * pDst,
          uint32_t blockSize)
{
    switch(S->iirType)
    {
        case ARM_IIR_DF1:
#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
        if(S->simdFlag)
            arm_iir_2nd_df1_simd_f32(S, pSrc, pDst, blockSize);
        else
#endif
        {
            /* In-place */
            if(pSrc == pDst)
                arm_biquad_cascade_df1_f32((arm_biquad_casd_df1_inst_f32 *)S, pSrc, pSrc, blockSize);
            else
                arm_biquad_cascade_df1_f32((arm_biquad_casd_df1_inst_f32 *)S, pSrc, pDst, blockSize);
        }
        break;

        case ARM_IIR_DF2:
        /* Not implemented */
        break;

        case ARM_IIR_DF1T:
        /* Not implemented */
        break;

        case ARM_IIR_DF2T: 
        /* In-place */
        if(pSrc == pDst)
            arm_biquad_cascade_df2T_f32((arm_biquad_cascade_df2T_instance_f32 *)S, pSrc, pSrc, blockSize);
        else
            arm_biquad_cascade_df2T_f32((arm_biquad_cascade_df2T_instance_f32 *)S, pSrc, pDst, blockSize);
        break;
    }
}

