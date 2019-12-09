/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_iir_generic_f32.c
 * Description:  Generic IIR filter processing function
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
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
#include "arm_helium_utils.h"
#endif

static void arm_iir_generic_df1_f32(
  const arm_iir_instance_f32 * S, 
        float32_t * pSrc, 
        float32_t * pDst, 
        uint32_t blockSize)
{
    uint32_t sample;
    float32_t x0, y0;
    uint32_t nbCascaded = S->numStages;
    uint16_t order = S->order;
    float32_t * pState = S->pState;
    float32_t * pCoeffs = S->pCoeffs;
    float32_t *pIn  = pSrc;
    float32_t * pOut;

    /* In-place */
    if(pSrc == pDst)
        pOut = pSrc;
    else
        pOut = pDst;

    while(nbCascaded > 0)
    {
        sample = blockSize;

        while (sample > 0U)
        {
            /* Load x[n] */
            x0 = *pIn++;

            /* Compute y[n] */
            y0 = pCoeffs[0]*x0;
            for(int i=0; i<order; i++)
            {
                y0 += pCoeffs[i+1]*pState[i] + pCoeffs[order+i+1]*pState[order+i];
            }

            /* Store y[n] */
            *pOut++ = y0;

            /* Update state */
            for(int i=order; i>1; i--)   
            {
                pState[i-1] = pState[i-2];
                pState[order+i-1] = pState[order+i-2];
            }
            pState[0] = x0;
            pState[order] = y0;
    
            sample--;
        }

        pState += 2*order;

        pOut-=blockSize;
        pIn = pOut;

        pCoeffs += 2*order+1;   

        nbCascaded--;
    }
}

#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
static void arm_iir_generic_df1_simd_f32(
  const arm_iir_instance_f32 * S, 
        float32_t * pSrc, 
        float32_t * pDst, 
        uint32_t blockSize)
{
    uint16_t order = S->order;
    uint32_t sample;
    float32_t x0, y0;
    uint8_t nbCascaded = S->numStages;
    float32_t * pState = S->pState;
    float32_t * pCoeffs = S->pCoeffs;
    float32_t * pIn = pSrc;
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

    /* x[n-1] ... x[n-order] */
    float32x4_t value_xm[order];
    /* y[n-1] ... y[n-order] */
    float32x4_t value_ym[order];

    /* Coefficients */
    float32x4_t coeff_xp3;
    float32x4_t coeff_xp2;
    float32x4_t coeff_xp1;
    float32x4_t coeff_x0;
    float32x4_t coeff_xm[order];
    float32x4_t coeff_ym[order];

    while(nbCascaded > 0)
    {
        /* Load coefficients */
        coeff_xp3 = vld1q_f32(pCoeffs+0 );
        coeff_xp2 = vld1q_f32(pCoeffs+4 );
        coeff_xp1 = vld1q_f32(pCoeffs+8 );
        coeff_x0  = vld1q_f32(pCoeffs+12);
        for(int i=0; i<order; i++)
        {
            coeff_xm[i] = vld1q_f32(pCoeffs+16+4*i);
            coeff_ym[i] = vld1q_f32(pCoeffs+16+4*order+4*i);
            /* Load x[n-1] ... x[n-order] */
            value_xm[i] = vdupq_n_f32(pState[i]);
            /* Load y[n-1] ... y[n-order] */
            value_ym[i] = vdupq_n_f32(pState[order+i]);
        }

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
            y0123 = vmlaq_f32 (y0123, coeff_xp2, value_xp2);
            y0123 = vmlaq_f32 (y0123, coeff_xp1, value_xp1);
            y0123 = vmlaq_f32 (y0123, coeff_x0 , value_x0 );
            for(int i=0; i<order; i++)
            {
                y0123 = vmlaq_f32 (y0123, coeff_xm[i], value_xm[i] );
                y0123 = vmlaq_f32 (y0123, coeff_ym[i], value_ym[i] );
            }
#endif
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
            y0123 = vfmaq_f32 (y0123, coeff_xp2, value_xp2);
            y0123 = vfmaq_f32 (y0123, coeff_xp1, value_xp1);
            y0123 = vfmaq_f32 (y0123, coeff_x0 , value_x0 );
            for(int i=0; i<order; i++)
            {
                y0123 = vfmaq_f32 (y0123, coeff_xm[i], value_xm[i] );
                y0123 = vfmaq_f32 (y0123, coeff_ym[i], value_ym[i] );
            }

#endif
            /* Store y[n] y[n+1] y[n+2] y[n+3] */
            vst1q_f32(pOut, y0123);
            pOut += 4;

            /* Update state */
            for(int i=order-1; i>3; i--)
            {
                value_xm[i] = value_xm[i-4];
                value_ym[i] = value_ym[i-4];
            }
            value_xm[3] = value_x0;
            value_xm[2] = value_xp1;
            value_xm[1] = value_xp2;
            value_xm[0] = value_xp3;
            value_ym[3] = vdupq_n_f32(y0123[0]);
            value_ym[2] = vdupq_n_f32(y0123[1]);
            value_ym[1] = vdupq_n_f32(y0123[2]);
            value_ym[0] = vdupq_n_f32(y0123[3]);

            sample--;
        }
   
        /* Tail */
        sample = blockSize & 0x3U;

        while (sample > 0U)
        {
            /* Load x[n] */
            x0 = *pIn++;

            /* Compute y[n] */
            y0 = pCoeffs[12]*x0;
            for(int i=0; i<order; i++)
                y0 += pCoeffs[16+i*4]*(value_xm[i])[0] + pCoeffs[16+4*(order)+4*i]*(value_ym[i])[0];

            /* Store y[n] */
            *pOut++ = y0;

            /* Update state */
            for(int i=order; i>1; i--) 
            {
                (value_xm[i-1])[0] = (value_xm[i-2])[0];
                (value_ym[i-1])[0] = (value_ym[i-2])[0];
            }
            (value_xm[0])[0] = x0;
            (value_ym[0])[0] = y0;
 
            sample--;
        }

        /* Store the updated state variables back into the pState array */
        for(int i=0; i<order; i++)
        {
            pState[i] = (value_xm[i])[0];
            pState[order+i] = (value_ym[i])[0];
        }

        pState +=2*order;

        pOut-=blockSize;
        pIn = pOut;

        pCoeffs += 4*(4+2*order);
        nbCascaded--;
    }
}
#endif 

/**
   * @brief  Processing function for the floating-point IIR filter of any order.
   * @param[in]     S          points to an instance of the floating-point IIR filter.
   * @param[in]     pSrc       points to the block of input data.
   * @param[out]    pDst       points to the block of output data.
   * @param[in]     blockSize  number of samples to process.
 */

void arm_iir_generic_f32(
  const arm_iir_instance_f32 * S, 
        float32_t * pSrc, 
        float32_t * pDst, 
        uint32_t blockSize)
{
#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
    if(S->simdFlag)
        arm_iir_generic_df1_simd_f32(S, pSrc, pDst, blockSize);
    else
#endif
        arm_iir_generic_df1_f32(S, pSrc, pDst, blockSize);
}

