/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_exponential_smoothing_f32.c
 * Description:  Floating-point exponential smoothing 
 *
 * $Date:        14th November, 2019
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
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
#include "arm_helium_utils.h"
#endif
#if defined(ARM_ERROR_HANDLER)
#include "arm_error.h"
#endif

/**
  @ingroup groupFilters
 */

/**
  @defgroup ExponentialSmoothings

  This set of functions implements the exponential smoothing.

  The exponential smoothing is a technique based on a 1st order IIR filter often used for 
  smoothing of time-series data. While in the simple moving average the past 
  observations are weighted equally, with the exponential smoothing an 
  exponentially decreasing weight is assigned.

  @par           Algorithm

  @par
  <pre>
      s1[n] = a*x1[n] + (1-a)*s1[n-1]
      ...                               where 0 < a < 1
      sk[n] = a*xk[n] + (1-a)*sk[n-1]
  </pre>
  a (alpha) is called smoothing factor. Larger values of a reduce the 
  smmothing effect and are more responsive to recent changes.

  @par           Usage
  The source array must contain the samples in the following order:
  <pre>
      x1[1], ..., xk[1], x1[2], ..., xk[2], ..., x1[N], ..., xk[N]
  </pre>
  .
  The destination array will contain s1[n] ... sk[n].
  
  @par
  It's important that the size of the source vector is a multiple of 
  the size of the destination vector (k x N).
 */

/**
  @addtogroup ExponentialSmoothings Vector exponential smoothing
  @{
 */

/**
 * @param[in]  S          points to an instance of the exponential smoothing structure.
 * @param[in]  pSrc       points to the block of input data.
 * @param[out] pDst       points to the block of output data.
 * @param[in]  blockSize  number of samples to process.
 */

void arm_exponential_smoothing_f32(arm_exp_smooth_instance_f32* S, const float32_t * pSrc, float32_t * pDst, uint32_t blockSize)
{
    uint32_t sample;
    float32_t alpha = S->alpha;
    uint32_t vecSize = S->vectorSize;
    uint32_t countVec;

#if defined(ARM_ERROR_HANDLER)
    if(blockSize%vecSize!=0)
        arm_error_handler(ARM_ERROR_MATH, "Source size is not a multiple of destination size.");
#endif

    /* Initialize number of vectors */
    countVec = blockSize/vecSize;

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
    f32x4_t Xn, Yn;
    f32x4_t vb0 = vdupq_n_f32(alpha);
    f32x4_t va1 = vdupq_n_f32(1-alpha);
    f32x4_t pState;

    while(countVec > 0)
    {
        /* Compute 4 samples at a time */
        sample = vecSize >> 2;

        while (sample > 0U)
        {
            /* Load sk[n-1] */
            pState = vld1q_f32(pDst);
            /* Load xk[n] */
            Xn = vld1q_f32(pSrc);
            pSrc+=4;

            /* Compute sx[n] and store it */
            Yn = vmulq_f32(Xn, vb0);
            Yn = vfmaq_f32(Yn, pState, va1);
            vst1q_f32(pDst, Yn);

            pDst+=4;

            sample--;
        }

        /* Tail */
        sample = vecSize & 0x3U;

        if (sample > 0U)
        {
            mve_pred16_t p0 = vctp32q(sample);

            pState = vld1q_f32(pDst);
            Xn = vld1q_f32(pSrc);
            Yn = vmulq_f32(Xn, vb0);
            Yn = vfmaq_f32(Yn, pState, va1);
            vstrwq_p(pDst, Yn, p0);
            
            pDst+=sample;
            pSrc+=sample;
        }

        pDst -= vecSize;
        countVec--;
    }

#else    

    float32_t x0, y0, ym1;
#if defined(ARM_MATH_NEON)
    float32x4_t Xn, Yn;
    float32x4_t vb0 = vdupq_n_f32(alpha);
    float32x4_t va1 = vdupq_n_f32(1-alpha);
    float32x4_t pState;
#endif

    while(countVec > 0)
    {
#if defined(ARM_MATH_NEON)
        /* Compute 4 samples at a time */
        sample = vecSize >> 2U;

        while (sample > 0U)
        {
            /* Load sk[n-1] */
            pState = vld1q_f32(pDst);
            /* Load xk[n] */
            Xn = vld1q_f32(pSrc);
            pSrc+=4;

            /* Compute sx[n] and store it */
            Yn = vmulq_f32(Xn, vb0);
            Yn = vmlaq_f32(Yn, pState, va1);
            vst1q_f32(pDst, Yn);

            pDst+=4;

            sample--;
        }

        /* Tail */
        sample = vecSize & 0x3U;
#else
        sample = vecSize;
#endif

        while (sample > 0U)
        {
            ym1 = *pDst;
            x0  = *pSrc++;

            y0 = alpha*x0 + (1-alpha)*ym1;

            *pDst = y0;

            pDst++;

            sample--;
        }

        pDst -= vecSize;
        countVec--;
    }

#endif /* if defined(ARM_MATH_MVEF) */
}

/**
  @} end of ExponentialSmoothings group
 */

