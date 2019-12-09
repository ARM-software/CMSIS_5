/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_iir_3rd_f32.c
 * Description:  3rd order IIR filter processing function
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

static void arm_iir_3rd_df1_f32(
    const arm_iir_instance_f32 * S,
          float32_t * pSrc,
          float32_t * pDst,
          uint32_t blockSize)
{
    float32_t *pIn = pSrc;                   /*  source pointer            */
    float32_t *pOut;                         /*  destination pointer       */
    float32_t *pState = S->pState;           /*  pState pointer            */
    float32_t *pCoeffs = S->pCoeffs;         /*  coefficient pointer       */
    float32_t acc;                           /*  Simulates the accumulator */
    uint32_t sample, stage = S->numStages;   /*  loop counters             */
    float32_t Xns;

#ifndef ARM_MATH_NEON
    float32_t xm1, xm2, xm3, ym1, ym2, ym3;
    float32_t b0, b1, b2, b3, a1, a2, a3;
#endif

    /* In-place */
    if(pSrc == pDst)
        pOut = pSrc;
    else
        pOut = pDst;

#if defined(ARM_MATH_NEON)
    float32x4_t Xn;
    float32x4_t Yn;
    float32x4_t a;
    float32x4_t b;
    
    float32x4_t x,tmp;
    float32x4_t t;
    float32x2_t tsum;
#endif

    while (stage > 0U)
    {
#ifdef ARM_MATH_NEON
        /* Load state */
        Xn = vld1q_f32(pState);
        Xn[3] = 0;
        Xn = vrev64q_f32(Xn);
        Xn = vcombine_f32(vget_high_f32(Xn), vget_low_f32(Xn));
    
        Yn = vld1q_f32(pState + 3);
        Yn[3] = 0;
        Yn = vrev64q_f32(Yn);
        Yn = vcombine_f32(vget_high_f32(Yn), vget_low_f32(Yn));
    
        /* Load coefficients */
        b = vld1q_f32(pCoeffs);
        b = vrev64q_f32(b);  
        b = vcombine_f32(vget_high_f32(b), vget_low_f32(b));
    
        a = vld1q_f32(pCoeffs + 4);
        a = vrev64q_f32(a);
        a = vcombine_f32(vget_high_f32(a), vget_low_f32(a));
        a[0] = 0.0;
    
        pCoeffs += 7;

        /* Compute 4 outputs at a time */
        sample = blockSize >> 2U;
    
        while (sample > 0U)
        {
            /* Read the first 4 inputs */
            x = vld1q_f32(pIn);
      
            pIn += 4;
            tmp = vextq_f32(Xn, x, 1);
            t = vmulq_f32(b, tmp);
      
            t = vmlaq_f32(t, a, Yn);
            tsum = vpadd_f32(vget_high_f32(t),vget_low_f32(t));
            tsum = vpadd_f32(tsum, tsum);
            t = vcombine_f32(tsum, tsum);
            Yn = vextq_f32(Yn, t, 1);
      
            tmp = vextq_f32(Xn, x, 2);
            t = vmulq_f32(b, tmp);
            t = vmlaq_f32(t, a, Yn);
            tsum = vpadd_f32(vget_high_f32(t),vget_low_f32(t));
            tsum = vpadd_f32(tsum, tsum);
            t = vcombine_f32(tsum, tsum);
            Yn = vextq_f32(Yn, t, 1); 
      
            tmp = vextq_f32(Xn, x, 3);
            t = vmulq_f32(b, tmp);
            t = vmlaq_f32(t, a, Yn);
            tsum = vpadd_f32(vget_high_f32(t),vget_low_f32(t));
            tsum = vpadd_f32(tsum, tsum);
            t = vcombine_f32(tsum, tsum);
            Yn = vextq_f32(Yn, t, 1);
      
            Xn = x;
            t = vmulq_f32(b, Xn);
            t = vmlaq_f32(t, a, Yn);
            tsum = vpadd_f32(vget_high_f32(t),vget_low_f32(t));
            tsum = vpadd_f32(tsum, tsum);
            t = vcombine_f32(tsum, tsum);
            Yn = vextq_f32(Yn, t, 1);
           
            /* Store the 4 outputs and increment the pointer */
            vst1q_f32(pOut, Yn);
            pOut += 4;

            /* Decrement the loop counter */
            sample--;
        }
    
        /* Tail */
        sample = blockSize & 0x3U;

#else
        /* Load state */
        xm1 = pState[0]; 
        xm2 = pState[1];
        xm3 = pState[2];
        ym1 = pState[3];
        ym2 = pState[4];
        ym3 = pState[5];
    
        /* Load coefficients */
        b0 = *pCoeffs++;
        b1 = *pCoeffs++;
        b2 = *pCoeffs++;
        b3 = *pCoeffs++;
        a1 = *pCoeffs++;
        a2 = *pCoeffs++;
        a3 = *pCoeffs++;
    
        sample = blockSize;
#endif

        while (sample > 0U)
        {
            /* Read the input */
            Xns = *pIn++;

            /* y[n] =  b0*x[n] + b1*x[n-1] + b2*x[n-2] + b3*x[n-3] + a1*y[n-1] + a2*y[n-2] + a3*y[n-3] */
#ifdef ARM_MATH_NEON
            acc = (b[3]*Xns) + (b[2]*Xn[3]) + (b[1]*Xn[2]) + (b[0]*Xn[1]) + (a[3]*Yn[3]) + (a[2]*Yn[2]) + (a[1]*Yn[1]);
#else
            acc = (b0*Xns) + (b1*xm1) + (b2*xm2) + (b3*xm3) + (a1*ym1) + (a2*ym2) + (a3*ym3);
#endif

            /* Store the result in the accumulator in the destination buffer. */
            *pOut++ = acc;
      
            /* Update the state */
#ifdef ARM_MATH_NEON
            Xn[1] = Xn[2];
            Xn[2] = Xn[3];
            Xn[3] = Xns;
            Yn[1] = Yn[2];
            Yn[2] = Yn[3];
            Yn[3] = acc;
#else
            xm3 = xm2;
            xm2 = xm1;
            xm1 = Xns;
            ym3 = ym2;
            ym2 = ym1;
            ym1 = acc;
#endif

            /* Decrement the loop counter */
            sample--;
        }

#ifdef ARM_MATH_NEON
        Xn = vrev64q_f32(Xn);
        Xn = vcombine_f32(vget_high_f32(Xn), vget_low_f32(Xn) );
        vst1q_f32(pState, Xn);
        pState += 3;
        Yn = vrev64q_f32(Yn);
        Yn = vcombine_f32(vget_high_f32(Yn), vget_low_f32(Yn) );
        Yn[3] = 0;
        vst1q_f32(pState, Yn );
        pState += 3;
#else
        *pState++ = xm1;
        *pState++ = xm2;
        *pState++ = xm3;
        *pState++ = ym1;
        *pState++ = ym2;
        *pState++ = ym3;
#endif
        pOut-=blockSize;
        pIn = pOut;

        /* Decrement the loop counter */
        stage--;
    }
}

#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
static void arm_iir_3rd_df1_simd_f32(
    const arm_iir_instance_f32 * S,
          float32_t * pSrc,
          float32_t * pDst,
          uint32_t blockSize)
{
    uint32_t sample;
    float32_t x0, xm1, xm2, xm3, y0, ym1, ym2, ym3;
    float32_t b0, b1, b2, b3, a1, a2, a3;
    uint32_t nbCascaded = S->numStages;
    float32_t * pState = S->pState;
    float32_t * pCoeffs = S->pCoeffs;
    float32_t *pIn  = pSrc;
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

    /* x[n-3] x[n-3] x[n-3] x[n-3] */
    float32x4_t value_xm3;
    /* x[n-2] x[n-2] x[n-2] x[n-2] */
    float32x4_t value_xm2;
    /* x[n-1] x[n-1] x[n-1] x[n-1] */
    float32x4_t value_xm1;
    /* y[n-3] y[n-3] y[n-3] y[n-3] */
    float32x4_t value_ym3;
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
    float32x4_t coeff_xm3;
    float32x4_t coeff_ym1;
    float32x4_t coeff_ym2;
    float32x4_t coeff_ym3;

    while(nbCascaded > 0)
    {
        /* Load coefficients */
        b0 = pCoeffs[12];
        b1 = pCoeffs[16];
        b2 = pCoeffs[20];
        b3 = pCoeffs[24];

        a1 = pCoeffs[28];
        a2 = pCoeffs[32];
        a3 = pCoeffs[36];

        /* Load x[n-1], x[n-2], x[n-3], y[n-1], y[n-2], y[n-3] */
        xm1 = pState[0];
        xm2 = pState[1];
        xm3 = pState[2];
        ym1 = pState[3];
        ym2 = pState[4];
        ym3 = pState[5];

        /* Coefficient vectors */
        coeff_xp3 = vld1q_f32(pCoeffs+0 );
        coeff_xp2 = vld1q_f32(pCoeffs+4 );
        coeff_xp1 = vld1q_f32(pCoeffs+8 );
        coeff_x0  = vld1q_f32(pCoeffs+12);
        coeff_xm1 = vld1q_f32(pCoeffs+16);
        coeff_xm2 = vld1q_f32(pCoeffs+20);
        coeff_xm3 = vld1q_f32(pCoeffs+24);
        coeff_ym1 = vld1q_f32(pCoeffs+28);
        coeff_ym2 = vld1q_f32(pCoeffs+32);
        coeff_ym3 = vld1q_f32(pCoeffs+36);
   
        /* Create vectors for x[n-3], x[n-2], x[n-1], y[n-3], y[n-2], y[n-1] */
        value_xm3 = vdupq_n_f32(xm3);
        value_xm2 = vdupq_n_f32(xm2);
        value_xm1 = vdupq_n_f32(xm1);
        value_ym3 = vdupq_n_f32(ym3);
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
    
            y0123 = vmulq_f32 (coeff_xp3, value_xp3);
#if defined(ARM_MATH_NEON)
            y0123 = vmlaq_f32(y0123, coeff_xp2, value_xp2);
            y0123 = vmlaq_f32(y0123, coeff_xp1, value_xp1);
            y0123 = vmlaq_f32(y0123, coeff_x0 , value_x0 );
            y0123 = vmlaq_f32(y0123, coeff_xm1, value_xm1);
            y0123 = vmlaq_f32(y0123, coeff_xm2, value_xm2);
            y0123 = vmlaq_f32(y0123, coeff_xm3, value_xm3);
            y0123 = vmlaq_f32(y0123, coeff_ym1, value_ym1);
            y0123 = vmlaq_f32(y0123, coeff_ym2, value_ym2);
            y0123 = vmlaq_f32(y0123, coeff_ym3, value_ym3);
#endif
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            y0123 = vfmaq_f32(y0123, coeff_xp2, value_xp2);
            y0123 = vfmaq_f32(y0123, coeff_xp1, value_xp1);
            y0123 = vfmaq_f32(y0123, coeff_x0 , value_x0 );
            y0123 = vfmaq_f32(y0123, coeff_xm1, value_xm1);
            y0123 = vfmaq_f32(y0123, coeff_xm2, value_xm2);
            y0123 = vfmaq_f32(y0123, coeff_xm3, value_xm3);
            y0123 = vfmaq_f32(y0123, coeff_ym1, value_ym1);
            y0123 = vfmaq_f32(y0123, coeff_ym2, value_ym2);
            y0123 = vfmaq_f32(y0123, coeff_ym3, value_ym3);
#endif

            /* Store y[n] y[n+1] y[n+2] y[n+3] */
            vst1q_f32(pOut, y0123);
            pOut += 4;
    
            /* Update state */
            value_xm3 = value_xp1; /* x[n+1] -> x[n-3] */
            value_xm2 = value_xp2; /* x[n+2] -> x[n-2] */
            value_xm1 = value_xp3; /* x[n+3] -> x[n-1] */
            value_ym3 = vdupq_n_f32(y0123[1]); /* y[n+1] -> y[n-3] */
            value_ym2 = vdupq_n_f32(y0123[2]); /* y[n+2] -> y[n-2] */
            value_ym1 = vdupq_n_f32(y0123[3]); /* y[n+3] -> y[n-1] */
    
            sample--;
        }
    
        /* Tail */
        sample = blockSize & 0x3U;
    
        /* Load state */
        xm3 = value_xp1[0];
        xm2 = value_xp2[0];
        xm1 = value_xp3[0];

        ym3 = y0123[1];
        ym2 = y0123[2];
        ym1 = y0123[3];

        while (sample > 0U)
        {
            x0 = *pIn++;
            
            y0 = b0*x0 + b1*xm1 + b2*xm2 + b3 * xm3 + a1*ym1 + a2*ym2 + a3*ym3;

            *pOut++ = y0;
            
            /* Update state */
            xm3 = xm2;
            xm2 = xm1;
            xm1 = x0;

            ym3 = ym2;
            ym2 = ym1;
            ym1 = y0;
    
            sample--;
        }
    
        /* Store the updated state variables back into the pState array */
        pState[0] = xm1;
        pState[1] = xm2;
        pState[2] = xm3;
        pState[3] = ym1;
        pState[4] = ym2;
        pState[5] = ym3;

        pState += 6;

        pOut-=blockSize;
        pIn = pOut;

        pCoeffs += 40;
 
        nbCascaded--;
    }
}
#endif 

/**
   * @brief  Processing function for the floating-point IIR filter of 3rd order.
   * @param[in]     S          points to an instance of the floating-point IIR filter.
   * @param[in]     pSrc       points to the block of input data.
   * @param[out]    pDst       points to the block of output data.
   * @param[in]     blockSize  number of samples to process.
 */

void arm_iir_3rd_f32(
    const arm_iir_instance_f32 * S,
          float32_t * pSrc,
          float32_t * pDst,
          uint32_t blockSize)
{
#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
    if(S->simdFlag)
        arm_iir_3rd_df1_simd_f32(S, pSrc, pDst, blockSize);
    else
#endif
        arm_iir_3rd_df1_f32(S, pSrc, pDst, blockSize);
}
