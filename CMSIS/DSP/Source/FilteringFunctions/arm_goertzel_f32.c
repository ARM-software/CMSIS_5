/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_goertzel_f32.c
 * Description:  Floating-point Goertzel DFT function
 *
 * $Date:        2019
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

/**
  @ingroup groupFilters
 */

/**
  @defgroup Goertzel_DFTs 

  This set of functions implements Goertzel DFTs
  for Q15, Q31 and floating-point data types.

  @par  Introduction
  The Goertzel algorithm is used to evaluate the individual terms of 
  the discrete Fourier transform (DFT) of a discrete signal. It has been described by
  American theoretical physicist Gerald Goertzel in 1958.

  @par
  The Goertzel algorithm has the form of a cascade of two filters that take as input the 
  sequence of N samples x[n] and returns the output y[n]. The first stage is
  an IIR filter and it is used to compute the intermediate sequence s[n], the second stage
  is a FIR filter:
  .
  <pre>
      s[n] = x[n] + 2*cos[w0]*s[n-1] - s[n-2]
      y[n] = s[n] - exp(-j*w0)*s[n-1]
  </pre>
  .
  where w0 is the frequency to analyse expressed in radians per sample.

  @par
  The equivalent time-domain sequence is the following summation of n+1 terms:
  .
  <pre>
      y[n] = x[0]*exp(j*w0*(n-0)) + x[1]*exp(j*w0*(n-1)) + ... +  x[n]*exp(j*w0*(n-n))
  </pre>

  @par
  If the frequencies are restricted to the form w0 = 2*pi*k/N, where k = 0, ..., N-1:
  .
  <pre>
      y[N] = x[0]*exp(j*2*pi*k/N*(N-0)) + x[1]*exp(j*2*pi*k/N*(N-1)) + ... +  x[N]*exp(j*2*pi*k/N*(N-N)) =
           = x[0]*exp(-j*2*pi*k*(0)/N) + x[1]*exp(-j*2*pi*k*(1)/N) + ... +  x[N] =
           = X[k] + x[N]
  </pre>
  .
  where X[k] is the DFT of the sequence x[n].

  @par
  To compute the previous summation, N+1 terms are needed, but only N input terms are available and x[N] is
  missing. The sequence is extended with x[N]=0 and y[N] correspond to the DFT X[k].

  @par Usage
  The initialization function \ref arm_goertzel_init_f32() is used to store the value of 
  w0 in a specific instance. For the Neon function a working array of size 8 is 
  needed and it must be allocated by the user and linked to the same instance at init time.
 
  @par
  The output to the Goertzel DFT is an array of 2 elements, the real part and the imaginary part of the DFT
  evaluated for w0.
 */

/**
  @addtogroup Goertzel_DFTs Vector Goertzel DFT
  @{
 */

/**
  @brief         Processing function for the Goertzel DFT.
  @param[in]     S          points to an instance of the floating-point Goertzel DFT structure
  @param[in]     pSrc       points to the block of input data
  @param[out]    pDst       points to the block of output data
  @param[in]     blockSize  number of samples to process
  @return        none
 */

void arm_goertzel_f32(
        arm_goertzel_instance_f32 *S,
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize)
{       
    uint32_t blkCnt;                         /* Temporary variables for counts */
    float32_t cosine, sine;
    float32_t sample, sn, sn_1, sn_2;
    float32_t acc_re, acc_im;
    
    cosine = S->cosine;
    sine   = S->sine;

    /* == IIR: s[n] = x[n] + 2*cos[w0]*s[n-1] - s[n-2] == */

    /* Initialize s[n-2] and s[n-1] */
    sn_2 = 0;
    sn_1 = 0;

#if defined(ARM_MATH_NEON) || ( defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) )
    /* 
     * c = 2cos(w0)
     * 
     *        |  s(4k)  |        
     *        | s(4k+1) |         | s(4k-1) |         |  x(4k)  |
     * s(n) = | s(4k+2) | = [ C ] | s(4k-2) | + [ D ] | x(4k+1) |   for   0 < k < N/4
     *        | s(4k+3) |                             | x(4k+2) |
     *
     * where:
     *
     *         |     c         -1    |             |   1       0    0  0 |
     *         |   c^2-1       -c    |             |   c       1    0  0 |
     * [ C ] = |   c^3-2c     1-c^2  |  ;  [ D ] = | c^2-1     c    1  0 |
     *         | c^4-3c^2+1   2c-c^3 |             | c^3-2c  c^2-1  c  1 |
     *
     *             coeff1     coeff2                 coeff3  coeff4 c5 c6
     */

    float32x4_t s, x; 
    float32x4_t coeff1, coeff2, coeff3, coeff4, coeff5, coeff6;
    float32_t * pCoeffs;

    /* Initialise coefficient vectors */  
    pCoeffs = S->coeffs;
    coeff6 = vld1q_f32(pCoeffs);
    coeff5 = vld1q_f32(pCoeffs+1);
    coeff4 = vld1q_f32(pCoeffs+2);
    coeff3 = vld1q_f32(pCoeffs+3);
    coeff2 = vnegq_f32(coeff3);
    coeff1 = vld1q_f32(pCoeffs+4);

    /* Initialise loop count */
    blkCnt = blockSize >> 2;
  
    while (blkCnt > 0U)
    {
        /* Load [ x(n) x(n+1) x(n+2) x(n+3) ] */
        x = vld1q_f32(pSrc);

        /* Compute [ y(n) y(n+1) y(n+2) y(n+3) ] */        
        s = vmulq_n_f32(coeff1, sn_1);
#if defined(ARM_MATH_NEON)
        s = vmlaq_n_f32(s, coeff2, sn_2);
        
        s = vmlaq_n_f32(s, coeff3, x[0]);
        s = vmlaq_n_f32(s, coeff4, x[1]);
        s = vmlaq_n_f32(s, coeff5, x[2]);
        s = vmlaq_n_f32(s, coeff6, x[3]);
#else
        s = vfmaq_n_f32(s, coeff2, sn_2);

        s = vfmaq_n_f32(s, coeff3, x[0]);
        s = vfmaq_n_f32(s, coeff4, x[1]);
        s = vfmaq_n_f32(s, coeff5, x[2]);
        s = vfmaq_n_f32(s, coeff6, x[3]);
#endif

        pSrc+=4;

        /* Update state: s[n-1] -> s[n-2] ; s[n] -> s[n-1] */
        sn_1 = s[3];
        sn_2 = s[2];
    
        /* Decrement counter */
        blkCnt--;
    }
  
    /* Tail */
    blkCnt = blockSize & 0x3;
#else
    /* Initialise loop count */
    blkCnt = blockSize;
#endif

    while (blkCnt > 0U)
    {
        /* Load x(n) */
        sample = *pSrc++;

        /* Compute s(n) = x(n) + 2*cos(w0)*s(n-1) - s(n-2) */ 
        sn = sample + 2*cosine * sn_1 - sn_2;

        /* Update state: s[n-1] -> s[n-2] ; s[n] -> s[n-1] */
        sn_2 = sn_1;
        sn_1 = sn;

        /* Decrement counter */
        blkCnt--;
    }

    /* x[N] = 0 -> s[N] = 2*cos[w0]*s[n-1] - s[n-2] */
    sn = 2*cosine * sn_1 - sn_2;

    /* == FIR: y[n] = s[n] - (cos[w0] - j*sin[w0])*s[n-1] = Re{y[n]} + j*Im{y[n]} == */

    /* Re{y[n]} = s[n] - cos[w0]*s[n-1] */
    acc_re = sn - cosine *sn_1;
    /* Im{y[n]} = sin[w0]*s[n-1] */
    acc_im = sine*sn_1;

    /* Divide by scaling factor */
    acc_re *= (S->scaling);
    acc_im *= (S->scaling);

    /* Store result */
    *pDst++ = acc_re;
    *pDst++ = acc_im;
}

/**
  @} end of Goertzel_DFTs group
 */
